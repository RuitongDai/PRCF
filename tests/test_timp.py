"""Check that AMP windows never bridge an episode reset or a routing change."""
from importlib.util import find_spec
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import torch
from mjlab.sensor import ObjRef
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_euler_xyz, yaw_quat

if find_spec("prcf.tasks.parkour.config.TIMP") is None:
    raise unittest.SkipTest("TIMP is unavailable in this checkout")

from prcf.tasks.parkour.config.TIMP.mdp.observations import AMP_DIM, AmpRayCastCfg
from prcf.tasks.parkour.config.TIMP.rl.algorithm import TimpPPO, WindowReplay
from prcf.tasks.parkour.config.TIMP.rl.models import HISTORY, PROPRIO_DIM
from prcf.tasks.parkour.config.TIMP.rl.motion_loader import ReferenceClip, ReferencePools, WINDOW
from prcf.tasks.parkour.config.TIMP.terrains import HEADS, amp_head, amp_heads_for, e1_amp_parkour_terrain_cfg


class TimpTests(unittest.TestCase):
    def test_amp_heads_follow_terrain_columns(self):
        generator = e1_amp_parkour_terrain_cfg()
        names = tuple(generator.sub_terrains)
        box_cols = [names.index("high_box"), names.index("long_high_box")]
        terrain = SimpleNamespace(
            terrain_types=torch.tensor([*range(len(names)), *box_cols]),
            cfg=SimpleNamespace(terrain_generator=generator),
        )
        env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain), device="cpu")
        actual = amp_head(env)
        expected = ("locomotion", "locomotion", "locomotion", "stairs", "stairs", "stairs",
                    "box", "box", "box", "box")
        self.assertEqual(actual.tolist(), [HEADS.index(name) for name in expected])
        self.assertEqual(amp_heads_for(("long_high_box", "high_box")), ("box",))

    def test_shared_box_pool_preserves_each_reset_edge(self):
        pools = object.__new__(ReferencePools)
        pools.device = "cpu"
        motion = torch.zeros(WINDOW, AMP_DIM)
        state = torch.zeros(WINDOW, 13)
        box = {"size_xyz": (1.2, 1.2, 0.9), "position": (2.0, 0.1, 0.45)}
        pools.clips = {"box": [
            ReferenceClip(motion, state, 1.0, {}, box, "front"),
            ReferenceClip(motion, state, 1.0, {}, box, "rear"),
        ]}
        tiles = e1_amp_parkour_terrain_cfg().sub_terrains
        for terrain in ("high_box", "long_high_box"):
            tile = tiles[terrain]
            (ascent, descent), weights = pools.box_reset_pool(terrain, tile, 0.9)
            torch.testing.assert_close(ascent[:, 0], torch.full((WINDOW,), tile.front_x - 1.4))
            torch.testing.assert_close(descent[:, 0], torch.full((WINDOW,), tile.end_x - 2.6))
            torch.testing.assert_close(weights, torch.ones(2))
            self.assertEqual(pools.box_reset_pool(terrain, tile, 0.7)[0], [])

    def test_amp_scan_uses_export_heading_at_steep_pitch(self):
        quaternion = quat_from_euler_xyz(torch.tensor([0.7]), torch.tensor([math.radians(88)]),
                                         torch.tensor([-0.4]))
        sensor = AmpRayCastCfg(name="scan", frame=ObjRef(type="body", name="pelvis")).build()
        actual = sensor._extract_yaw_rotation(matrix_from_quat(quaternion))
        expected = matrix_from_quat(yaw_quat(quaternion))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)

    def test_route_change_and_auto_reset_clear_amp_window(self):
        actor_obs = torch.zeros(1, HISTORY, PROPRIO_DIM)
        algorithm = object.__new__(TimpPPO)
        vars(algorithm).update(
            history=torch.zeros(1, WINDOW, AMP_DIM), last_head=torch.full((1,), -1),
            window_length=torch.zeros(1, dtype=torch.long), amp_statistics=torch.zeros(3),
            replay={head: WindowReplay(5, "cpu") for head in (0, 2)},
            discriminator=SimpleNamespace(reward=lambda windows, head: windows[:, -1, 0]),
            stage=2, stage_step=0, him_pairs=[], device="cpu",
            transition=SimpleNamespace(observations={"actor": actor_obs, "critic": torch.zeros(1, 3)}),
        )

        def step(frame, head, done=False):
            obs = {"amp": torch.full((1, AMP_DIM), float(frame)),
                   "amp_route": torch.tensor([[head]]), "actor": actor_obs}
            algorithm.process_env_step(obs, torch.zeros(1), torch.tensor([done]), {})

        with patch("rsl_rl.algorithms.PPO.process_env_step") as parent:
            for frame in range(1, 5):
                step(frame, 0)
            self.assertEqual(algorithm.replay[0].count, 1)
            self.assertEqual(parent.call_args.args[1].item(), 4.0)
            step(5, 0, done=True)
            step(6, 0)
            step(7, 2)
            step(8, 2)
            step(9, 2)
            self.assertEqual(parent.call_args.args[1].item(), 0.0)
            step(10, 2)
            self.assertEqual(algorithm.replay[0].count, 1)
            self.assertEqual(algorithm.replay[2].count, 1)
            torch.testing.assert_close(algorithm.replay[2].buffer[0, :, 0], torch.arange(7.0, 11.0))
