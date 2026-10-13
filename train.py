"""Train PPO on the navigation scene with random spawns and save reproducible run artifacts."""
import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from raycast_env import RaycastEnv


class RolloutRewardCallback(BaseCallback):
    """Select checkpoints by mean return over a full window of training episodes."""

    def __init__(self, output, window=100):
        super().__init__()
        if window < 1:
            raise ValueError('window must be positive')
        self.output = Path(output)
        self.episode_rewards = deque(maxlen=window)
        self.best_checkpoint_reward = -np.inf

    def _on_step(self):
        # Monitor reports complete episode returns, including failures and timeouts.
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done and 'episode' in info:
                self.episode_rewards.append(float(info['episode']['r']))
        return True

    def _on_rollout_end(self):
        count = len(self.episode_rewards)
        self.logger.record('rollout/best_selection_episodes', count)
        if count < self.episode_rewards.maxlen:
            return
        mean_reward = float(np.mean(self.episode_rewards))
        self.logger.record('rollout/best_selection_mean_reward', mean_reward)
        if mean_reward > self.best_checkpoint_reward:
            # This callback runs before PPO updates the policy for this rollout.
            self.model.save(self.output / 'best_model')
            self.best_checkpoint_reward = mean_reward
            (self.output / 'best_model_metrics.json').write_text(json.dumps({
                'metric': 'rollout_mean_episode_reward',
                'mean_reward': mean_reward, 'episodes': count,
                'timesteps': self.num_timesteps,
            }, indent=2) + '\n')
            print(f'New best rollout mean reward: {mean_reward:.2f} ({count} episodes)')
        self.logger.record('rollout/best_checkpoint_mean_reward', self.best_checkpoint_reward)


class TrainingViewer(BaseCallback):
    """Display the actual training simulator, rather than a separate policy demo."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.viewer = None
        self.last_frame = None

    def _on_training_start(self):
        # Import GUI dependencies only when visual training is requested.
        import mujoco
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.env.model.camera('top').id
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        print('Live training: close the window to stop and save the current policy.')

    def _on_step(self):
        from view_scene import draw_rays

        if not self.viewer.is_running():
            return False  # SB3 exits learn(); main then saves final_model.zip.
        # Pace collection at roughly real time. PPO optimization and periodic
        # evaluation pause the picture because they do not step this simulator.
        if self.last_frame is not None:
            remaining = self.env.frame_skip * self.env.model.opt.timestep - (time.perf_counter() - self.last_frame)
            if remaining > 0:
                time.sleep(remaining)
        # SB3 automatically resets completed episodes before this callback.
        # Recast to show the current pose, including after an automatic reset.
        geomids, distances = self.env.sensor.perform_raycast(self.env.robot_id, self.env.ray_length)
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            draw_rays(self.viewer.user_scn, self.env.sensor, self.env.robot_id,
                      geomids, distances, ray_length=self.env.ray_length)
        self.viewer.sync()
        self.last_frame = time.perf_counter()
        return True

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=300_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-envs', type=int, default=4, help='Independent training environments')
    parser.add_argument('--vec-env', choices=('dummy', 'subproc'), default='subproc',
                        help='Sequential environments or separate worker processes')
    parser.add_argument('--best-window', type=int, default=100,
                        help='Completed training episodes required to select the best model')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--render', action='store_true', help='Watch training in real time')
    parser.add_argument('--stage', choices=RaycastEnv.STAGES, default='hard')
    parser.add_argument('--resume', type=Path, help='Continue training a saved PPO model')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Override the discount factor, including when resuming')
    args = parser.parse_args()
    if args.best_window < 1:
        parser.error('--best-window must be positive')
    if args.n_envs < 1:
        parser.error('--n-envs must be positive')
    if args.render and args.vec_env == 'subproc':
        parser.error('--render requires --vec-env dummy to access the live simulator')
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.gamma is not None and not 0 < args.gamma <= 1:
        parser.error('--gamma must be in (0, 1]')
    if args.resume is not None and not args.resume.is_file():
        parser.error('--resume must point to an existing model.zip file')
    Train(args).run()


class Train:
    """Train one PPO policy using independent vectorized MuJoCo simulators."""

    def __init__(self, args):
        self.args = args

    def make_envs(self, evaluation=False, monitor_dir=None):
        args = self.args
        # Evaluation stays in the parent process; training can use worker processes.
        vec_class = DummyVecEnv if evaluation or args.vec_env == 'dummy' else SubprocVecEnv
        return make_vec_env(
            RaycastEnv, n_envs=1 if evaluation else args.n_envs,
            seed=args.seed + args.n_envs if evaluation else args.seed,
            env_kwargs={'stage': args.stage},
            monitor_dir=str(monitor_dir) if monitor_dir is not None else None,
            vec_env_cls=vec_class,
            vec_env_kwargs={'start_method': 'spawn'} if vec_class is SubprocVecEnv else None,
        )

    def run(self):
        args = self.args
        output = args.output or Path(__file__).parent / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S')
        # Refuse existing directories to avoid overwriting earlier models or logs.
        output.mkdir(parents=True, exist_ok=False)
        torch.set_num_threads(1)  # Small networks run efficiently on a single CPU thread.
        env = self.make_envs(monitor_dir=output)
        evaluation = None
        training_viewer = None
        try:
            evaluation = self.make_envs(evaluation=True)
            training_viewer = TrainingViewer(env.envs[0].unwrapped) if args.render else None
            # Evaluation remains diagnostic; only training rollouts select the best.
            callback = EvalCallback(evaluation, best_model_save_path=None,
                                    log_path=str(output), eval_freq=max(1, min(10_000, max(2048, args.steps)) // args.n_envs),
                                    n_eval_episodes=10, deterministic=True)
            rollout_callback = RolloutRewardCallback(output, window=args.best_window)
            if args.resume:
                # Restore policy and optimizer, attaching a fresh environment at the
                # selected difficulty. Do not reinitialize the learned network.
                # Pass the override while loading so the rollout buffer and policy
                # algorithm use the same discount factor from initialization.
                overrides = {} if args.gamma is None else {'gamma': args.gamma}
                model = PPO.load(args.resume, env=env, device='cpu', seed=args.seed, **overrides)
            else:
                model = PPO('MlpPolicy', env, seed=args.seed, device='cpu', verbose=1, tensorboard_log="./tensorboard_logs/",
                            n_steps=2048, batch_size=64, learning_rate=3e-4,
                            gamma=0.999 if args.gamma is None else args.gamma, ent_coef=0.01)
            print(f'Training stage={args.stage}, gamma={model.gamma}')
            # Custom model metadata travels with both best and final checkpoints.
            model.curriculum_stage = args.stage
            (output / 'config.json').write_text(json.dumps({
                'requested_steps': args.steps, 'seed': args.seed,
                'algorithm': 'PPO', 'scene': 'random robot and target positions',
                'stage': args.stage,
                'gamma': model.gamma,
                'n_envs': args.n_envs, 'vec_env': args.vec_env,
                'n_steps_per_env': model.n_steps,
                'rollout_size': model.n_steps * args.n_envs,
                'best_model_metric': 'rollout_mean_episode_reward',
                'best_model_window': args.best_window,
                'resume': str(args.resume.resolve()) if args.resume else None,
            }, indent=2) + '\n')
            # PPO collects complete rollouts, so actual steps may exceed the request.
            callbacks = [rollout_callback, callback]
            if training_viewer:
                callbacks.insert(0, training_viewer)
            model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name="raycast_training",
                        reset_num_timesteps=args.resume is None)
            model.save(output / 'final_model')
            if len(rollout_callback.episode_rewards) < args.best_window:
                print(f'Best model not selected: fewer than {args.best_window} completed episodes. '
                      'The final model is available as final_model.zip.')
            print(f'Models and logs saved in: {output.resolve()}')
        finally:
            if training_viewer is not None:
                training_viewer.close()
            env.close()
            if evaluation is not None:
                evaluation.close()


if __name__ == '__main__':
    main()
