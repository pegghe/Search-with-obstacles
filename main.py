"""Evaluate a saved PPO agent without opening a viewer."""
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from raycast_env import RaycastEnv


def evaluate(env, policy, episodes, seed):
    successes, collisions, timeouts, distances, returns, durations = [], [], [], [], [], []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        successes.append(info["is_success"])
        collisions.append(info["collision"])
        timeouts.append(info["termination_reason"] == "timeout")
        distances.append(info["target_distance"])
        returns.append(total_reward)
        durations.append(env.data.time)
    return (f"success rate={np.mean(successes):.1%}, "
            f"collision rate={np.mean(collisions):.1%}, "
            f"timeout rate={np.mean(timeouts):.1%}, "
            f"final distance={np.mean(distances):.3f} m, "
            f"mean reward={np.mean(returns):.2f}, "
            f"mean duration={np.mean(durations):.2f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["eval"])
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to best_model.zip or final_model.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=RaycastEnv.STAGES,
                        help="Override the difficulty saved in the model")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    stage = args.stage or getattr(model, "curriculum_stage", "hard")
    env = RaycastEnv(stage=stage)
    try:
        model.set_env(env)
        print(f"Evaluation stage: {stage}; episodes: {args.episodes}; seed: {args.seed}")
        print("PPO agent:    ", evaluate(env, model, args.episodes, args.seed))
    finally:
        env.close()


if __name__ == "__main__":
    main()
