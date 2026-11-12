"""Viewing must not create persistent runs or alter a training checkpoint."""
from contextlib import chdir,redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
from test_mimic import fixture_motion


class PlayTests(unittest.TestCase):
    def test_play_uses_temporary_reference_and_cleans_up(self):
        from scripts.play import main
        _,motion=fixture_motion()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'input.npz';np.savez_compressed(source,**motion)
            env=Mock();runner=Mock();used=[]
            def check_during_viewer():
                cfg=build.call_args.args[1]
                reference=Path(cfg.commands['motion'].motion_file)
                self.assertTrue(reference.exists())
                used.append(reference)
                self.assertIsNone(build.call_args.kwargs.get('log_dir'))
            with chdir(root),patch('scripts.play.build_runner',return_value=(env,runner)) as build,patch('mjlab.viewer.NativeMujocoViewer') as viewer:
                viewer.return_value.run.side_effect=check_during_viewer
                self.assertEqual(main(['--motion-file',str(source),'--checkpoint','model.pt','--device','cpu']),0)
            env.close.assert_called_once()
            self.assertEqual(len(used),1)
            self.assertFalse(used[0].exists())
            self.assertFalse((root/'logs').exists())
            self.assertFalse((root/'outputs').exists())

    def test_play_dry_run_leaves_no_run_directory(self):
        from scripts.play import main
        _,motion=fixture_motion()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.npz';np.savez_compressed(source,**motion)
            with chdir(root),redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--motion-file',str(source),'--dry-run']),0)
            self.assertEqual([p.name for p in root.iterdir()],['input.npz'])


if __name__=='__main__':unittest.main()
