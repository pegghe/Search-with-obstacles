# Confronto di tutte le run

Valutazione: ambiente RaycastEnv attuale, stage hard, 100 episodi, seed 42–141, policy deterministica. Tutti i checkpoint disponibili sono stati rivalutati.

| Run | Checkpoint | Step | Successi | Collisioni | Timeout | Reward media | Distanza finale (m) | Durata (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hard-rays | best_model.zip | 790000 | 42% | 38% | 20% | -2.62 | 2.058 | 14.89 |
| hard-rays | final_model.zip | 1001472 | 38% | 45% | 17% | -4.58 | 2.089 | 14.10 |
| hard-rays-2obstacles | best_model.zip | 1291472 | 65% | 30% | 5% | 10.21 | 1.475 | 7.47 |
| hard-rays-2obstacles | final_model.zip | 1392528 | 74% | 20% | 6% | 14.69 | 1.315 | 7.70 |
| hard-rays-2obstacles1 | best_model.zip | 1431472 | 72% | 25% | 3% | 13.68 | 1.331 | 6.75 |
| hard-rays-2obstacles1 | final_model.zip | 1592528 | 59% | 19% | 22% | 7.52 | 1.599 | 13.60 |
| hard-rays-2obstacles2 | best_model.zip | 1711472 | 73% | 21% | 6% | 14.47 | 1.269 | 7.81 |
| hard-rays-2obstacles2 | final_model.zip | 1933232 | 69% | 17% | 14% | 12.33 | 1.430 | 10.48 |
| hard-rays-2obstacles3 | best_model.zip | 1771472 | 78% | 15% | 7% | 16.46 | 1.265 | 8.26 |
| hard-rays-plus300k | best_model.zip | 1091472 | 59% | 32% | 9% | 5.51 | 1.855 | 9.87 |
| hard-rays-plus300k | final_model.zip | 1302528 | 54% | 36% | 10% | 2.51 | 1.967 | 11.15 |
| vectorized1 | best_model.zip | 960000 | 80% | 15% | 5% | 17.47 | 1.229 | 7.41 |
| vectorized1 | final_model.zip | 1007616 | 79% | 14% | 7% | 16.99 | 1.278 | 7.88 |
| vectorized2mln | best_model.zip | 1177616 | 74% | 18% | 8% | 14.58 | 1.395 | 7.93 |
| vectorized2mln | final_model.zip | 2015232 | 81% | 12% | 7% | 17.90 | 1.202 | 8.41 |
| vectorized2mlnCurriculum | best_model.zip | 1529040 | 79% | 18% | 3% | 16.11 | 1.328 | 7.54 |
| vectorized2mlnCurriculum | final_model.zip | 2299088 | 75% | 22% | 3% | 14.94 | 1.317 | 6.79 |

## Configurazioni salvate

### hard-rays

```json
{
  "requested_steps": 1000000,
  "seed": 42,
  "algorithm": "PPO",
  "scene": "random robot and target positions",
  "stage": "hard",
  "gamma": 0.999,
  "resume": null
}
```

### hard-rays-2obstacles

```json
{
  "requested_steps": 300000,
  "seed": 42,
  "algorithm": "PPO",
  "scene": "random robot and target positions",
  "stage": "hard",
  "gamma": 0.999,
  "resume": "/home/rl/raycast_update_copy/runs/hard-rays-plus300k/best_model.zip"
}
```

### hard-rays-2obstacles1

```json
{
  "requested_steps": 300000,
  "seed": 42,
  "algorithm": "PPO",
  "scene": "random robot and target positions",
  "stage": "hard",
  "gamma": 0.999,
  "resume": "/home/rl/raycast_update_copy/runs/hard-rays-2obstacles/best_model.zip"
}
```

### hard-rays-2obstacles2

```json
{
  "requested_steps": 500000,
  "seed": 42,
  "algorithm": "PPO",
  "scene": "random robot and target positions",
  "stage": "hard",
  "gamma": 0.999,
  "best_model_metric": "success_rate",
  "resume": "/home/rl/raycast_update_copy/runs/hard-rays-2obstacles1/best_model.zip"
}
```

### hard-rays-2obstacles3

```json
{
  "requested_steps": 1000000,
  "seed": 42,
  "algorithm": "PPO",
  "scene": "random robot and target positions",
  "stage": "hard",
  "gamma": 0.999,
  "best_model_metric": "success_rate",
  "resume": "/home/rl/raycast_update_copy/runs/hard-rays-2obstacles2/best_model.zip"
}
```

### hard-rays-plus300k

```json
{
  "requested_steps": 300000,
  "seed": 42,
  "algorithm": "PPO",
  "scene": "random robot and target positions",
  "stage": "hard",
  "gamma": 0.999,
  "resume": "/home/rl/raycast_update/runs/hard-rays/final_model.zip"
}
```

### vectorized1

```json
{
  "requested_steps": 1000000,
  "seed": 42,
  "algorithm": "PPO",
  "scene": "random robot and target positions",
  "stage": "hard",
  "gamma": 0.999,
  "n_envs": 4,
  "vec_env": "subproc",
  "n_steps_per_env": 2048,
  "rollout_size": 8192,
  "best_model_metric": "mean_reward",
  "resume": null
}
```

### vectorized2mln

```json
{
  "requested_steps": 1000000,
  "seed": 42,
  "algorithm": "PPO",
  "scene": "random robot and target positions",
  "stage": "hard",
  "gamma": 0.999,
  "n_envs": 4,
  "vec_env": "subproc",
  "n_steps_per_env": 2048,
  "rollout_size": 8192,
  "best_model_metric": "mean_reward",
  "resume": "/home/rl/raycast_update_copy/runs/20261013-113623/final_model.zip"
}
```

### vectorized2mlnCurriculum

```json
{
  "requested_steps": 1000000,
  "seed": 42,
  "algorithm": "PPO",
  "scene": "random robot and target positions",
  "stage": "hard",
  "gamma": 0.999,
  "n_envs": 4,
  "vec_env": "subproc",
  "n_steps_per_env": 2048,
  "rollout_size": 8192,
  "best_model_metric": "rollout_mean_episode_reward",
  "best_model_window": 100,
  "resume": "/home/rl/raycast_update_copy/runs/hard-rays-2obstacles/best_model.zip"
}
```

## Limiti e provenienza

- hard-rays-2obstacles3 non ha final_model.zip: non è stato valutato un final per questa run.
- Il percorso sorgente di vectorized2mln (runs/20261013-113623/final_model.zip) non è presente. Non è verificabile che coincida con vectorized1/final_model.zip.
- hard-rays-plus300k proviene da un checkpoint nella directory sorella raycast_update, non dalla cartella hard-rays di questa directory.
- Per le vecchie run senza best_model_metric, il timestep del best coincide con la valutazione a reward media massima; questo è un indizio, non una registrazione esplicita del criterio.
- I file dichiarano tutti stage hard. I nomi delle cartelle non permettono di ricostruire modifiche storiche alla scena.
- Gli step sono i contatori dei checkpoint, non la somma di tutti gli esperimenti della catena.
- I risultati su 100 episodi confrontano questi checkpoint e questi scenari, non stabiliscono la superiorità generale di un metodo.
