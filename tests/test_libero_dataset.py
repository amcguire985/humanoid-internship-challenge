"""Recorder contracts without launching MuJoCo."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from record_libero_dataset import RecordingEnv
from inspect_libero_dataset import load_transition


class RecorderTests(unittest.TestCase):
    def test_action_pairing_terminal_observation_and_vertical_flip(self):
        class Sim:
            model = SimpleNamespace(camera_names=['agentview', 'robot0_eye_in_hand'])
            data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7), time=0.)
            def render(self, width, height, camera_name):
                image = np.full((height, width, 3), self.data.time * 20, dtype=np.uint8)
                image[0] += 10
                return image
        class Env:
            def __init__(self):
                self.env = SimpleNamespace(sim=Sim(), robots=[SimpleNamespace(
                    _ref_joint_pos_indexes=np.arange(7), _ref_joint_vel_indexes=np.arange(7))])
                self.sent = []
            def check_success(self):
                return self.env.sim.data.time >= .1
            def obs(self):
                return dict(robot0_gripper_qpos=np.zeros(2), robot0_gripper_qvel=np.zeros(2),
                            robot0_eef_pos=np.full(3, self.env.sim.data.time), robot0_eef_quat=[0, 0, 0, 1])
            def step(self, action):
                self.sent.append(action.copy())
                self.env.sim.data.time += .05
                return self.obs(), 0., False, {}
        with tempfile.TemporaryDirectory() as d, h5py.File(Path(d) / 'episode.h5', 'w') as f:
            env = Env()
            wrapped = RecordingEnv(env, f, 'episode_001', 4)
            wrapped.configure_rendering = lambda: None
            wrapped.runner = SimpleNamespace(phase='PREGRASP', records=[])
            wrapped.step(np.zeros(7))  # Setup is deliberately outside the dataset.
            self.assertNotIn('actions', f)
            wrapped.start(env.obs())
            a = np.arange(7) / 10
            wrapped.step(a)
            wrapped.runner.phase = 'RELEASE_SETTLE'
            wrapped.step(-a)
            f.attrs.update(language_instruction='test task', episode_id='episode_001', episode_success=True)
            self.assertEqual(f['actions'].shape, (2, 7))
            self.assertEqual(f['observations/agentview_rgb'].shape, (3, 4, 4, 3))
            sample = load_transition(f, 0)
            np.testing.assert_array_equal(sample['action'], env.sent[1])
            self.assertAlmostEqual(sample['observation']['sim_time'], .05)
            self.assertAlmostEqual(sample['next_observation']['sim_time'], .1)
            self.assertEqual(sample['observation']['agentview_rgb'][0, 0, 0], 1)
            self.assertEqual(sample['observation']['agentview_rgb'][-1, 0, 0], 11)
            self.assertEqual(f['phase'][1], b'RELEASE_SETTLE')
            self.assertTrue(sample['success_after_action'])


if __name__ == '__main__':
    unittest.main()
