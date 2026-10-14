"""Watch a PPO agent with the standard Gymnasium MuJoCo renderer."""
import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import check_for_correct_spaces

from raycast_env import RaycastEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, help='Path to best_model.zip or final_model.zip')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--headless', action='store_true', help='Evaluate without opening a window')
    parser.add_argument('--stage', choices=RaycastEnv.STAGES,
                        help='Override the difficulty saved in the model')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    torch.set_num_threads(1)
    agent = PPO.load(args.model, device='cpu')
    # Older checkpoints have no stage metadata and used the original hard scene.
    stage = args.stage or getattr(agent, 'curriculum_stage', 'hard')
    print(f'Evaluation stage: {stage}')
    env = RaycastEnv(stage=stage, render_mode=None if args.headless else "human",
                     show_rays=not args.headless)
    try:
        # Prediction works with one simulator even when training used several.
        check_for_correct_spaces(env, agent.observation_space, agent.action_space)
        successes = 0
        for episode in range(args.episodes):
            observation, _ = env.reset()
            total_reward = 0.0
            while True:
                action, _ = agent.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    successes += int(info['is_success'])
                    print(f"Episode {episode + 1}: {info['termination_reason']}, "
                          f"reward={total_reward:.2f}, distance={info['target_distance']:.3f} m, "
                          f"time={env.data.time:.2f} s")
                    break
        print(f'Successes: {successes}/{args.episodes} (random spawns each episode).')
    finally:
        env.close()


if __name__ == '__main__':
    main()
