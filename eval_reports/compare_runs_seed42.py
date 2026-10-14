"""Evaluate all available run checkpoints on the same 100 hard scenarios."""
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]


def evaluate_checkpoint(task):
    import sys
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from raycast_env import RaycastEnv

    torch.set_num_threads(1)
    run, checkpoint = task
    model = PPO.load(ROOT / 'runs' / run / checkpoint, device='cpu')
    env = RaycastEnv(stage='hard')
    episodes = []
    try:
        for seed in range(42, 142):
            obs, _ = env.reset(seed=seed)
            reward_sum = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                if terminated or truncated:
                    break
            episodes.append(dict(seed=seed, success=bool(info['is_success']),
                                 collision=bool(info['collision']),
                                 timeout=info['termination_reason'] == 'timeout',
                                 reward=reward_sum, distance=float(info['target_distance']),
                                 duration=float(env.data.time)))
    finally:
        env.close()
    result = dict(run=run, checkpoint=checkpoint, timesteps=int(model.num_timesteps),
                  n_envs=int(model.n_envs), episodes=100, stage='hard', seed=42)
    for key in ['success', 'collision', 'timeout', 'reward', 'distance', 'duration']:
        result[key] = float(np.mean([e[key] for e in episodes]))
    return result, episodes


if __name__ == '__main__':
    tasks = [(p.name, name) for p in sorted((ROOT / 'runs').iterdir()) if p.is_dir()
             for name in ['best_model.zip', 'final_model.zip'] if (p / name).is_file()]
    report = ROOT / 'eval_reports' / 'all_runs_seed42.json'
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(evaluate_checkpoint, t): t for t in tasks}
        for future in as_completed(futures):
            result, episodes = future.result()
            results.append(dict(summary=result, episodes=episodes))
            results.sort(key=lambda x: (x['summary']['run'], x['summary']['checkpoint']))
            report.write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)
    with report.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]['summary']))
        writer.writeheader()
        writer.writerows(r['summary'] for r in results)
