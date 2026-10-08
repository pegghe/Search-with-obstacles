"""Check checkpoint selection independently of policy reward."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from train import SuccessEvalCallback
from raycast_env import RaycastEnv


class SuccessCheckpointTests(unittest.TestCase):
    def test_success_selects_checkpoint_and_ties_keep_previous(self):
        with tempfile.TemporaryDirectory() as output:
            env = RaycastEnv()
            self.addCleanup(env.close)
            callback = SuccessEvalCallback(env, output, n_eval_episodes=10)
            callback.model = Mock()
            with patch('train.EvalCallback._on_step', return_value=True):
                callback.n_calls = callback.eval_freq
                for successes, reward, expected_saves in [(6, 100, 1), (8, -100, 2),
                                                           (7, 200, 2), (8, 300, 2)]:
                    callback._is_success_buffer = [True] * successes + [False] * (10 - successes)
                    callback.last_mean_reward = reward
                    self.assertTrue(callback._on_step())
                    self.assertEqual(callback.model.save.call_count, expected_saves)
            self.assertEqual(callback.best_success_rate, 0.8)
            callback.model.save.assert_called_with(Path(output) / 'best_model')
            self.assertIsNone(callback.best_model_save_path)

    def test_final_evaluation_uses_success(self):
        with tempfile.TemporaryDirectory() as output:
            env = RaycastEnv()
            self.addCleanup(env.close)
            callback = SuccessEvalCallback(env, output, n_eval_episodes=10)
            callback.model = Mock()
            callback.best_success_rate = 0.8

            def evaluate(*args, **kwargs):
                callback._is_success_buffer.extend([True] * 9 + [False])
                return -100, 0

            with patch('train.evaluate_policy', side_effect=evaluate):
                callback.evaluate_final()
            self.assertEqual(callback.best_success_rate, 0.9)
            callback.model.save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
