"""Evaluate a saved PPO agent, optionally showing its movement and ray sensor."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import torch
from stable_baselines3 import PPO

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
    env = RaycastEnv(stage=stage)
    try:
        # Supplying env also checks saved observation/action space compatibility.
        agent.set_env(env)
        env.reset()
        if args.headless:
            context = nullcontext(None)
        else:
            import mujoco.viewer
            from view_scene import draw_rays
            context = mujoco.viewer.launch_passive(env.model, env.data)
        successes = 0
        with context as viewer:
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = env.model.camera('top').id
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
            for episode in range(args.episodes):
                observation, _ = env.reset()
                total_reward = 0.0
                while True:
                    if viewer is not None and not viewer.is_running():
                        return
                    frame_start = time.perf_counter()
                    simulated_start = env.data.time
                    # The policy chooses both commands; there is no keyboard control.
                    action, _ = agent.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if viewer is not None:
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            draw_rays(viewer.user_scn, env.sensor, env.robot_id,
                                      env.sensor.raycast_geomids_buf[env.robot_id],
                                      env.sensor.raycast_distances_buf[env.robot_id],
                                      ray_length=env.ray_length)
                        viewer.sync()
                        # Only visual playback is paced; training runs at full speed.
                        remaining = env.data.time - simulated_start - (time.perf_counter() - frame_start)
                        if remaining > 0:
                            time.sleep(remaining)
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
