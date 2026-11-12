"""Scene edits must survive save/reload without touching motion or quality gates."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import mujoco
import numpy as np

from prcf.motion.kinematics import build_kinematic_motion
from prcf.motion.schema import load_clip
from prcf.motion.validation import e1_model, sha256
from prcf.tasks.mimic.replay import ReplayDocument, WebReplayEditor, discover_motions


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
        self.source_hash = sha256(self.npz)
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
        return ReplayDocument(self.json, self.model)

    def test_save_reload_backup_and_training_geometry(self):
        doc = self.document()
        original_bytes = self.json.read_bytes()
        doc.edit_obstacle('box', {'size_xyz': [1.2, .8, .9]}, keep_bottom=True)
        doc.edit_obstacle('box', {'position': [2.3, -.1, .45]})
        doc.save()
        loaded = load_clip(self.json)
        self.assertEqual(loaded.scene['obstacles'][0]['position'], [2.3, -.1, .45])
        self.assertEqual(loaded.quality.max_environment_penetration, .015)
        self.assertEqual(loaded.quality.contact_requirements[0].part, 'left_hand')
        self.assertEqual(loaded.motion_to_scene, self.payload['motion_to_scene'])
        payload = json.loads(self.json.read_text())
        self.assertEqual(payload['metadata']['note'], 'keep me')
        self.assertNotIn('kinematic_scene_check', payload['metadata'])
        self.assertFalse(payload['metadata']['physics_tracking_validated'])
        self.assertEqual(sha256(self.npz), self.source_hash)
        backups = list((self.root / '.scene_backups').glob('*.json'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original_bytes)
        # Compile the saved scene through the SAME builder used by training.
        from prcf.motion.scene import SceneBuilder
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
        doc = ReplayDocument(flat, self.model)
        name = doc.add_box()
        doc.save()
        self.assertEqual(load_clip(flat.with_suffix('.json')).scene['obstacles'][0]['id'], name)
        paths, initial = discover_motions(self.npz)
        self.assertEqual(initial, self.json)
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(p.suffix == '.json' for p in paths))
        self.assertEqual(sha256(self.npz), self.source_hash)

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
        editor.dispatch('play')
        editor.tick(.02)
        self.assertEqual(editor.document.frame, 2)
        editor.loop.value = False
        editor.tick(1.)
        self.assertEqual(editor.document.frame, 4)
        self.assertFalse(editor.playing)
        editor.dispatch('save')
        self.assertFalse(editor.document.dirty)
        self.assertEqual(load_clip(self.json).scene['obstacles'][0]['position'], [2.4, .2, .4])
        self.assertEqual(sha256(self.npz), self.source_hash)
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
