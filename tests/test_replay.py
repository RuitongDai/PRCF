"""Scene edits must survive save/reload without touching motion or quality gates."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from prcf.asset.robot.e1.e1_constants import e1_model
from prcf.asset.scene import SceneBuilder, load_clip
from scripts.csv_to_npz_e1 import build_kinematic_motion
from scripts.export_reference import write_trajectory
from scripts.replay_motion import ReplayDocument, WebReplayEditor, discover_motions, set_native_frame


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _, _ = e1_model()
        q = cls.model.key_qpos[0]
        cls.motion = build_kinematic_motion(cls.model, np.tile(q[:3], (5, 1)),
                                           np.tile(q[3:7], (5, 1)), np.tile(q[7:], (5, 1)), 50.)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.npz = self.root / 'climb.npz'
        np.savez_compressed(self.npz, **self.motion)
        self.source_hash = hashlib.sha256(self.npz.read_bytes()).hexdigest()
        self.json = self.root / 'climb.json'
        self.payload = {
            'schema_version': 1, 'clip_id': 'climb', 'robot_id': 'e1_25dof',
            'motion_file': 'climb.npz', 'fps': 50,
            'motion_to_scene': {'position': [0, 0, .008], 'quaternion_wxyz': [1, 0, 0, 0]},
            'scene': {'ground_height': 0, 'obstacles': [
                {'id': 'box', 'type': 'box', 'size_xyz': [1., 1., .6], 'position': [2., 0., .3]}]},
            'quality': {'max_environment_penetration': .015, 'contact_requirements': [
                {'part': 'left_hand', 'start_frame': 0, 'end_frame': 4, 'min_fraction': .7}]},
            'metadata': {'note': 'keep me', 'kinematic_scene_check': {'old': True}},
        }
        self.json.write_text(json.dumps(self.payload))

    def document(self):
        return ReplayDocument(self.json)

    def test_amp_export_replays_world_pose_and_embedded_scene(self):
        path = self.root / 'amp.npz'
        clip = load_clip(self.json)
        root_pos = np.array([[0.8, -0.5, 0.9], [1.2, -0.3, 1.0]])
        pose = Rotation.from_euler('ZYX', [[1.1, 0.2, -0.1], [1.5, -0.1, 0.2]])
        root_quat = pose.as_quat(scalar_first=True)
        root_quat[1] *= -1  # Equivalent recorded rotation with the opposite sign.
        root_velocity = np.array([[0.3, 0.4, 0.1], [0.4, 0.6, 0.2]])
        data = {
            'root_pos_w': root_pos, 'root_quat_w': root_quat,
            'root_lin_vel_w': root_velocity, 'root_ang_vel_w': root_velocity,
            'dof_pos': self.motion['joint_pos'][:2], 'dof_vel': self.motion['joint_vel'][:2],
            'body_pos_w': np.broadcast_to(root_pos[:, None, :], (2, 13, 3)),
            'contact_state': np.zeros((2, 6)), 'dof_torque': np.ones((2, 25)),
            'reference_frame': np.array([1, 2]),
        }
        write_trajectory(path, data, fps=50, clip=clip, joint_names=self.motion['joint_names'],
                         body_names=self.motion['body_names'][1:14], termination='motion_finished',
                         expected_frames=2)
        with np.load(path, allow_pickle=False) as archive:
            self.assertEqual(archive['amp_obs'].shape, (2, 357))
        doc = ReplayDocument(path)
        self.assertEqual(doc.scene, clip.scene)
        state = mujoco.MjData(self.model)
        set_native_frame(self.model, state, doc.motion, 1)
        np.testing.assert_allclose(state.qpos[:3], root_pos[1])
        np.testing.assert_allclose(state.qpos[3:7], root_quat[1])
        np.testing.assert_allclose(state.qpos[7:], data['dof_pos'][1])
        np.testing.assert_allclose(state.qvel[:3], root_velocity[1])
        np.testing.assert_allclose(state.qvel[3:6], pose[1].inv().apply(root_velocity[1]))
        set_native_frame(self.model, state, doc.motion, 0.5)
        np.testing.assert_allclose(state.qpos[:3], root_pos.mean(axis=0))
        midway = Rotation.from_quat(state.qpos[3:7], scalar_first=True)
        np.testing.assert_allclose((midway.inv() * Slerp([0, 1], pose)(0.5)).magnitude(), 0., atol=1e-7)

    def test_save_reload_backup_and_training_geometry(self):
        doc = self.document()
        original_bytes = self.json.read_bytes()
        doc.edit_obstacle('box', {'size_xyz': [1.2, .8, .9]}, keep_bottom=True)
        doc.edit_obstacle('box', {'position': [2.3, -.1, .45]})
        doc.save()
        loaded = load_clip(self.json)
        self.assertEqual(loaded.scene['obstacles'][0]['position'], [2.3, -.1, .45])
        self.assertEqual(loaded.motion_to_scene, self.payload['motion_to_scene'])
        payload = json.loads(self.json.read_text())
        self.assertEqual(payload['metadata']['note'], 'keep me')
        self.assertEqual(payload['quality'], self.payload['quality'])
        self.assertNotIn('kinematic_scene_check', payload['metadata'])
        self.assertFalse(payload['metadata']['physics_tracking_validated'])
        self.assertEqual(hashlib.sha256(self.npz.read_bytes()).hexdigest(), self.source_hash)
        backups = list((self.root / '.scene_backups').glob('*.json'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original_bytes)
        # Compile the saved scene through the SAME builder used by training.
        spec = mujoco.MjSpec()
        SceneBuilder(loaded.scene)(spec)
        model = spec.compile()
        np.testing.assert_allclose(model.geom('prcf_box').size, [.6, .4, .45])
        np.testing.assert_allclose(model.geom('prcf_box').pos, [2.3, -.1, .45])

    def test_invalid_edit_rolls_back_and_box_base_is_anchored(self):
        doc = self.document()
        before = deepcopy(doc.scene)
        with self.assertRaises(ValueError):
            doc.edit_obstacle('box', {'size_xyz': [1, 1, -1]}, keep_bottom=True)
        self.assertEqual(doc.scene, before)
        self.assertFalse(doc.dirty)
        doc.edit_obstacle('box', {'size_xyz': [1, 1, 1.]}, keep_bottom=True)
        self.assertAlmostEqual(doc.obstacle('box')['position'][2], .5)
        doc.restore()
        self.assertEqual(doc.scene, before)

    def test_save_detects_external_edits(self):
        doc = self.document()
        self.json.write_text(self.json.read_text() + '\n')
        doc.edit_obstacle('box', {'position': [3, 0, .3]})
        with self.assertRaisesRegex(RuntimeError, '其他程序'):
            doc.save()
        self.assertTrue(doc.dirty)

    def test_plain_npz_add_save_and_discovery_deduplicates(self):
        flat = self.root / 'walk.npz'
        np.savez_compressed(flat, **self.motion)
        doc = ReplayDocument(flat)
        name = doc.add_box()
        doc.save()
        self.assertEqual(load_clip(flat.with_suffix('.json')).scene['obstacles'][0]['id'], name)
        paths, initial = discover_motions(self.npz)
        self.assertEqual(initial, self.json)
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(p.suffix == '.json' for p in paths))
        self.assertEqual(hashlib.sha256(self.npz.read_bytes()).hexdigest(), self.source_hash)

    def test_stairs_parameters_and_rotation_validation(self):
        doc = self.document()
        scene = {'ground_height': 0, 'obstacles': [
            {'id': 'stairs', 'type': 'stairs', 'position': [1, 0, 0],
             'step_height': .15, 'step_depth': .3, 'width': 1., 'num_steps': 3}]}
        doc.replace_scene(scene)
        doc.edit_obstacle('stairs', {'step_height': .2, 'num_steps': 4})
        before = deepcopy(doc.scene)
        with self.assertRaisesRegex(ValueError, 'yaw'):
            doc.edit_obstacle('stairs', {'quaternion_wxyz': [.7071067811865476, .7071067811865476, 0, 0]})
        self.assertEqual(doc.scene, before)
        doc.save()
        self.assertEqual(load_clip(self.json).scene['obstacles'][0]['num_steps'], 4)

    def test_real_viser_gui_api_without_network_transport(self):
        # Use real Viser scene/GUI APIs and mjviser mesh conversion. Only the
        # websocket transport is replaced; this environment denies socket().
        import viser
        from viser._gui_api import GuiApi
        from viser._scene_api import SceneApi
        server = object.__new__(viser.ViserServer)
        server._scene_lifecycle_lock = threading.RLock()
        server._client_lock = threading.Lock()
        server._connected_clients = {}
        server._client_connect_cb = []
        server._websock_server = MagicMock()
        server._event_loop = MagicMock()
        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(pool.shutdown)
        server._thread_executor = pool
        server.scene = SceneApi(server, thread_executor=pool, event_loop=server._event_loop)
        server.gui = GuiApi(server, thread_executor=pool, event_loop=server._event_loop)
        flat = self.root / 'walk.npz'
        np.savez_compressed(flat, **self.motion)
        editor = WebReplayEditor(self.json, server=server)
        editor.dispatch('size', (1.1, .9, .8))
        editor.dispatch('position', (2.4, .2, .4))
        np.testing.assert_allclose(editor.box_handles['box'].dimensions, [1.1, .9, .8])
        editor.dispatch('next', 1)
        self.assertEqual(editor.path, flat)
        editor.dispatch('next', -1)
        self.assertTrue(editor.document.dirty)
        self.assertEqual(editor.document.obstacle('box')['position'], [2.4, .2, .4])
        editor.dispatch('step', 1)
        self.assertEqual(editor.document.frame, 1)
        roots = editor.document.motion['body_pos_w']
        roots[2, 0, 0] = roots[1, 0, 0] + 0.1
        editor.dispatch('play')
        editor.tick(.01)
        self.assertEqual(editor.document.frame, 1)
        self.assertAlmostEqual(editor.state.qpos[0], roots[1:3, 0, 0].mean())
        editor.tick(.01)
        self.assertEqual(editor.document.frame, 2)
        self.assertEqual(editor.frame_slider.value, 1)
        editor.loop.value = False
        editor.tick(1.)
        self.assertEqual(editor.document.frame, 4)
        self.assertEqual(editor.frame_slider.value, 4)
        self.assertFalse(editor.playing)
        editor.dispatch('save')
        self.assertFalse(editor.document.dirty)
        self.assertEqual(load_clip(self.json).scene['obstacles'][0]['position'], [2.4, .2, .4])
        self.assertEqual(hashlib.sha256(self.npz.read_bytes()).hexdigest(), self.source_hash)
        self.assertGreater(server._websock_server.queue_message.call_count, 20)
        # Exercise the actual wire serializer, not just Python handle updates.
        import msgspec.msgpack
        for call in server._websock_server.queue_message.call_args_list:
            msgspec.msgpack.encode(call.args[0].as_serializable_dict())
        self.assertTrue(editor.queue.empty())  # programmatic slider sync must not seek/pause
        editor._client_event(SimpleNamespace(client_id=0), 'seek', 3)
        self.assertEqual(editor.queue.get_nowait(), ('seek', 3))
        check = editor.document.check_current_frame()
        self.assertEqual(check['frame'], 4)
        self.assertGreaterEqual(check['max_penetration_m'], 0.)


if __name__ == '__main__':
    unittest.main()
