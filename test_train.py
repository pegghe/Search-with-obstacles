"""Check reward-based checkpoint selection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from train import RolloutRewardCallback, Train
from argparse import Namespace
import numpy as np
from raycast_env import RaycastEnv


class RewardCheckpointTests(unittest.TestCase):
    def make_callback(self, output):
        callback = RolloutRewardCallback(output, window=3)
        callback.model = Mock()
        callback.model.logger = Mock()
        return callback

    def complete(self, callback, rewards):
        callback.locals = {'dones': [True] * len(rewards),
                           'infos': [{'episode': {'r': r}} for r in rewards]}
        callback._on_step()
        callback._on_rollout_end()

    def test_full_window_required_and_incomplete_episodes_ignored(self):
        with tempfile.TemporaryDirectory() as output:
            callback = self.make_callback(output)
            callback.locals = {'dones': [False], 'infos': [{'episode': {'r': 999}}]}
            callback._on_step()
            self.complete(callback, [-10, -20])
            callback.model.save.assert_not_called()
            self.complete(callback, [-30])
            callback.model.save.assert_called_once_with(Path(output) / 'best_model')
            self.assertEqual(callback.best_checkpoint_reward, -20)
            self.assertTrue((Path(output) / 'best_model_metrics.json').is_file())

    def test_rolling_mean_ties_and_worse_windows_keep_previous(self):
        with tempfile.TemporaryDirectory() as output:
            callback = self.make_callback(output)
            self.complete(callback, [1, 2, 3])
            self.complete(callback, [1, 2, 3])
            self.assertEqual(callback.model.save.call_count, 1)
            self.complete(callback, [-10, -10, -10])
            self.assertEqual(callback.model.save.call_count, 1)
            self.complete(callback, [10, 10, 10])
            self.assertEqual(callback.model.save.call_count, 2)
            self.assertEqual(callback.best_checkpoint_reward, 10)


class VectorEnvironmentTests(unittest.TestCase):
    def test_independent_simulators_and_timeout_auto_reset(self):
        trainer = Train(Namespace(n_envs=2, vec_env='dummy', seed=42, stage='hard'))
        env = trainer.make_envs()
        self.addCleanup(env.close)
        observations = env.reset()
        self.assertEqual(observations.shape[0], 2)
        self.assertIsNot(env.envs[0].unwrapped.model, env.envs[1].unwrapped.model)
        self.assertFalse(np.array_equal(observations[0], observations[1]))
        for wrapped_env in env.envs:
            wrapped_env.unwrapped.max_episode_steps = 1
        observations, rewards, dones, infos = env.step(np.zeros((2, 2)))
        self.assertTrue(dones.all())
        for info in infos:
            self.assertTrue(info['TimeLimit.truncated'])
            self.assertIn('terminal_observation', info)


if __name__ == '__main__':
    unittest.main()
