"""Train PPO on the navigation scene with random spawns and save reproducible run artifacts."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from raycast_env import RaycastEnv


class SuccessEvalCallback(EvalCallback):
    """Save the highest success rate; keep the earlier checkpoint on ties."""

    def __init__(self, eval_env, output, **kwargs):
        # Disable EvalCallback's reward-based checkpoint saving.
        super().__init__(eval_env, best_model_save_path=None, **kwargs)
        self.output = Path(output)
        self.best_success_rate = -1.0

    def save_if_best(self):
        if len(self._is_success_buffer) != self.n_eval_episodes:
            raise RuntimeError('Evaluation must report is_success for every episode')
        success_rate = float(np.mean(self._is_success_buffer))
        if success_rate > self.best_success_rate:
            self.model.save(self.output / 'best_model')
            self.best_success_rate = success_rate
            print(f'New best success rate: {success_rate:.1%}')

    def _on_step(self):
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.save_if_best()
        return continue_training

    def evaluate_final(self):
        self._is_success_buffer = []
        evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                        deterministic=self.deterministic, callback=self._log_success_callback)
        print(f'Final policy success rate: {np.mean(self._is_success_buffer):.1%}')
        self.save_if_best()


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
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--render', action='store_true', help='Watch training in real time')
    parser.add_argument('--stage', choices=RaycastEnv.STAGES, default='hard')
    parser.add_argument('--resume', type=Path, help='Continue training a saved PPO model')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Override the discount factor, including when resuming')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.gamma is not None and not 0 < args.gamma <= 1:
        parser.error('--gamma must be in (0, 1]')
    if args.resume is not None and not args.resume.is_file():
        parser.error('--resume must point to an existing model.zip file')
    output = args.output or Path(__file__).parent / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S')
    # Refuse existing directories to avoid overwriting earlier models or logs.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)  # Small networks run efficiently on a single CPU thread.
    env = Monitor(RaycastEnv(stage=args.stage), str(output / 'train.monitor.csv'))
    evaluation = Monitor(RaycastEnv(stage=args.stage))
    evaluation.reset(seed=args.seed + 1)
    training_viewer = TrainingViewer(env.unwrapped) if args.render else None
    try:
        # Evaluation uses a separate simulator and deterministic actions.
        # Average evaluation across multiple random spawn pairs.
        callback = SuccessEvalCallback(evaluation, output=output,
                                log_path=str(output), eval_freq=min(10_000, max(2048, args.steps)),
                                n_eval_episodes=10, deterministic=True)
        if args.resume:
            # Restore policy and optimizer, attaching a fresh environment at the
            # selected difficulty. Do not reinitialize the learned network.
            # Pass the override while loading so the rollout buffer and policy
            # algorithm use the same discount factor from initialization.
            overrides = {} if args.gamma is None else {'gamma': args.gamma}
            model = PPO.load(args.resume, env=env, device='cpu', seed=args.seed, **overrides)
        else:
            model = PPO('MlpPolicy', env, seed=args.seed, device='cpu', verbose=1, tensorboard_log="./tensorboard_logs/",
                        n_steps=2048, batch_size=64, learning_rate=1e-4,
                        gamma=0.999 if args.gamma is None else args.gamma, ent_coef=0.01)
        print(f'Training stage={args.stage}, gamma={model.gamma}')
        # Custom model metadata travels with both best and final checkpoints.
        model.curriculum_stage = args.stage
        (output / 'config.json').write_text(json.dumps({
            'requested_steps': args.steps, 'seed': args.seed,
            'algorithm': 'PPO', 'scene': 'random robot and target positions',
            'stage': args.stage,
            'gamma': model.gamma,
            'best_model_metric': 'success_rate',
            'resume': str(args.resume.resolve()) if args.resume else None,
        }, indent=2) + '\n')
        # PPO collects complete rollouts, so actual steps may exceed the request.
        callbacks = [training_viewer, callback] if training_viewer else callback
        model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name="raycast_training",
                    reset_num_timesteps=args.resume is None)
        model.save(output / 'final_model')
        # Periodic callbacks run during rollout collection, before PPO updates
        # the network. Evaluate once more after the final update so a better
        # final policy is not missing from best_model.zip.
        callback.evaluate_final()
        print(f'Models and logs saved in: {output.resolve()}')
    finally:
        if training_viewer is not None:
            training_viewer.close()
        env.close()
        evaluation.close()


if __name__ == '__main__':
    main()
