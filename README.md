# raycast1

## Random obstacle dimensions

Training, evaluation and `watch_agent.py` keep the two obstacle centers fixed
at `(0, -1.4)` and `(0, 1.4)`, but independently randomize their width
(0.40–1.20 m) and depth (0.60–1.20 m) at every reset. Height stays 0.50 m.
Even at maximum size, the passages to the north/south walls are at least 1 m
wide and the gap between obstacles is at least 1.60 m. The robot, including
its head, fits inside a circle of diameter 0.84 m, leaving clearance to pass
and turn. Robot and target spawns use the sampled obstacle sizes.
The same reset seed reproduces sizes and spawns. `view_scene.py` shows the
static XML dimensions; the environment viewers show the random dimensions.
Older checkpoints can still be loaded, but their previous evaluation results
refer to fixed obstacle sizes and must be measured again on this environment.

A small MuJoCo project with a Gymnasium-controlled robot and a raycasting module
based on the code provided by the professor. Reinforcement learning will be
added in later steps.

## Installation

```bash
pip install -r requirements.txt
```

## Step 1: Preview the scene

Run from the `raycast1` directory in a desktop session:

```bash
python view_scene.py
```

The viewer starts with a top view of a 6 x 6 meter arena:

- Blue square: robot at `(-2, 0)`, facing +X.
- Light blue circle: head attached to the front of the robot.
- White dot: future ray origin, just outside the head.
- Orange rectangle: fixed obstacle at `(0, 0)`.
- Green cylinder: target at `(2, 0)`, visible to rays and physically passable.

Close the viewer window to exit. The scene is defined in `scene.xml`.
The viewer also displays the head-mounted ray sensor introduced in step 3.

## Step 2: Move the robot

Movement is controlled through `RaycastEnv.step([forward, turn])`.
`view_scene.py` is a static preview with no keyboard movement controls.

The robot has two translation joints (world X/Y) and one rotation joint (yaw).
The head is rigidly attached and turns with the robot. `movement.py` exposes
two normalized commands: forward speed and turn rate. It converts forward
speed into X/Y actuator targets using the current heading at every physics step.
Maximum speeds are 0.6 m/s and 1.2 rad/s. Three bounded velocity actuators
implement these two commands; this is an idealized planar drive, not a wheel model.

Walls and the obstacle physically block the robot during environment steps.
The target geometry has contacts disabled; the environment detects success by distance.

## Step 3: Head-mounted ray sensor

Run the same `python view_scene.py` command. Nine rays form a 120-degree fan
in front of the head, with a maximum range of 3 meters. Change `NRAYS`,
`FIELD_OF_VIEW`, and `RAY_LENGTH` in `view_scene.py` to adjust these settings.

- Orange rays stop at the first detected geometry (obstacle, arena wall, or target).
- Green rays extend to the maximum range when there is no hit.

In the environment, the sensor is updated after the physics steps and follows
the robot's current pose. The preview shows the rays at the initial pose. The origin is 1 cm in front of the head's
surface. With this forward-facing fan, rays point away from the robot and
do not hit its body. A wider fan pointing backward could detect the robot.
The target is a cylinder tall enough to intersect the horizontal rays. It has
`contype="0"` and `conaffinity="0"`, so it is detectable but physically passable.
The policy receives its class only when a ray hits it within range, with no
intervening geometry. A target behind the robot or beyond range is not reported.

Initially, the central ray hits the obstacle about 1.27 meters ahead.
The colored lines are rendering overlays only; they cannot affect physics or
sensor readings. Joint decorations are hidden at startup to avoid confusion.

## Step 4: Gymnasium environment

`raycast_env.py` defines `RaycastEnv`, combining the scene, controller, and sensor.
It runs without a viewer. The static scene preview remains available separately.
Install the updated requirements before using it.

```python
from raycast_env import RaycastEnv

env = RaycastEnv()
try:
    observation, info = env.reset(seed=42)
    for _ in range(25):
        # Move forward for 0.02 simulated seconds per action.
        observation, reward, terminated, truncated, info = env.step([1.0, 0.0])
        if terminated or truncated:
            break
finally:
    env.close()
```

Actions are `[forward, turn]` in `[-1, 1]`. The controller clips values to this
range. The environment assumes actions contain two valid numbers and that callers
call `reset()` before the first step and after each episode.

The observation is a float32 array with 27 values: 9 rays, each encoded as
`[distance, obstacle, target]`, flattened in ray order from
-60 to +60 degrees. Distance is divided by 3 m; 1 means maximum range or no hit.
The obstacle flag is 1 for obstacles and walls; the target flag is 1 for the
target. Both flags are 0 when nothing is detected. These flags are derived
from the geometry IDs returned by `raycast.py`.

Target coordinates and robot velocities are not included in the observation.
The simulator still uses the target position to compute rewards and success.
`info["target_distance"]` gives the planar distance from the robot center to the
target. Each reset samples new robot and target XY positions with zero velocity and
robot heading +X. Both positions have 0.47 m clearance from walls and obstacles,
and are at least 1 m apart. Reusing a seed reproduces the spawn; reset without
a seed continues the random sequence. Each observation owns its data and can be kept for later use.

## Step 5: Rewards and episode endings

Each action receives the sum of these components:

| Component | Value |
| --- | --- |
| Progress | `5 * (previous_distance - current_distance)` |
| Step | -0.01 per environment step, including the terminal step |
| Success | +20 when the robot center enters the target radius of 0.22 m |
| Collision | -20 when the base or head touches the obstacle or a wall |

Distances are measured from the robot center to the target in the XY plane.
Moving away gives negative progress; standing still receives only the step cost.
The cumulative step penalty after N actions is `-0.01 * N`, independent of
simulated duration and the number of physics substeps.
These are starting reward weights, to be evaluated during training.

Contacts are checked at every physics substep, including the final pose.
Floor contacts are ignored. Collision takes priority over simultaneous success.
Success or collision ends the episode with `terminated=True`; physics stops at
that substep. After 1000 actions (20 simulated seconds), an unfinished episode
ends with `truncated=True`. Timeout has no additional reward penalty.
Call `reset()` after either ending before calling `step()` again.

`info` includes `is_success`, `collision`, and `termination_reason` (`success`,
`collision`, `timeout`, or `None`). Step results also include `reward_components`
so each part of the reward can be inspected separately. Settings are defined
in `RaycastEnv.__init__`. No training algorithm is connected yet.

Run the interface and behavior checks with:

```bash
python -m unittest test_env -v
```

All observation values have finite bounds of `[0, 1]`.

## Raycasting usage

From a program in the `raycast1` directory, with MuJoCo `model` and `data` already created:

```python
import mujoco
from raycast import Raycaster

# Replace these names with body names from your MuJoCo scene.
robot_id = model.body("robot").id
indicator_id = model.body("robot_indicator").id
sensor = Raycaster(model, data, {robot_id: indicator_id})
sensor.setup_raycast(robot_id, nrays=9, angle_covered_degrees=90)

mujoco.mj_forward(model, data)
geomids, distances = sensor.perform_raycast(robot_id, ray_length=10)
```

The indicator defines the sensor's origin and orientation and can be the robot
body itself. Its local X-axis defines the forward direction.
Rays rotate around the world Z-axis, as in the original code.
Update poses with `mj_forward` or `mj_step` before each reading.

Each robot can have a different number of rays. Results contain the IDs of hit
geometries and distances in scene units (meters if the scene uses meters);
`-1` indicates no hit within `ray_length`.
Buffers are reused: use `.copy()` to retain a reading.

Compared with the original snippet, the class initializes all required structures,
uses unit directions, and sets `bodyexclude=-1` to include all bodies.
The robot itself can also be detected: place the indicator outside its geometry
to avoid self-intersections. The `1e-6` offset does not guarantee this.
The `mj_multiRay` signature was verified with MuJoCo 3.12.0.

## Step 6: Train and watch PPO

Models trained with the previous 9-, 14-, or 45-value observations are incompatible
with this 27-value observation. Start a new training run without `--resume`.

Install the updated dependencies, then start training from this directory:

```bash
pip install -r requirements.txt
python train.py --steps 300000 --output runs/ppo-first
```

PPO learns a neural policy that maps the 27 values from the 9 rays to forward and turning
commands. Training runs without a viewer on CPU. Robot and target positions are randomized at each reset. The reward settings are unchanged. Learning success is not guaranteed by
any particular number of steps; inspect evaluation outcomes before judging it.

To watch the robot and rays while PPO collects training experience:

```bash
python train.py --steps 300000 --output runs/ppo-live --render
```

This opens the actual training simulator and paces experience collection at
approximately real time, making training slower. The picture pauses during
network updates and separate evaluation episodes. Episodes reset automatically;
early movements are exploratory and may look erratic. Close the window to stop
training and save the current policy as `final_model.zip`. If stopped before the
first evaluation, `best_model.zip` will not yet exist. Use a desktop session and
a new output directory for each run. Omit `--render` for faster headless training.

Each new output directory contains:

- `best_model.zip`: highest evaluation reward seen during training.
- `final_model.zip`: policy after the final training update.
- `train.monitor.csv`: training episode returns and lengths.
- `evaluations.npz`: periodic evaluation rewards, lengths, and success flags.
- `config.json`: requested step count and seed.

Output directories must be new to protect previous runs. If `--output` is omitted,
training creates a timestamped directory in `runs`. PPO collects 2048-step rollouts,
so the actual step count can slightly exceed `--steps`. Evaluation runs every
10000 actions (or sooner for a short run). Best means highest reward, not guaranteed
success. Evaluation averages 10 episodes with random spawns and deterministic actions.

Evaluate a saved agent without opening a window, using the same command style as
the other projects:

```bash
python main.py eval --model runs/hard-rays-plus300k/best_model.zip --episodes 100 --seed 42
```

This reports success, collision, and timeout rates, mean final distance, mean
reward, and mean simulated episode duration. Episodes use random spawn positions;
the seed makes the evaluation reproducible. The difficulty defaults to the stage
saved in the model (hard for older checkpoints); use `--stage hard` to override it.
Training continues to use `train.py`.

Watch the saved agent with the rays visible:

```bash
python watch_agent.py runs/ppo-first/best_model.zip
```

This viewer displays the actual environment being controlled by the policy.
There are no manual movement commands. Close the window to exit. Terminal output
reports success, collision, or timeout, plus reward, final distance, and duration.
Use `--episodes 1` for one episode, or evaluate without a window:

```bash
python watch_agent.py runs/ppo-first/best_model.zip --headless --episodes 1
```

Evaluation episodes use new random spawn positions, so outcomes can differ even
with deterministic actions.
`view_scene.py` remains the static preview. A short plumbing check can be run with
`python train.py --steps 2048 --output runs/smoke-test-new`; this is not sufficient
evidence that the agent has learned navigation.

Implementation reference: [Stable-Baselines3 evaluation callbacks](https://stable-baselines3.readthedocs.io/en/v2.6.0/guide/callbacks.html).

## Progressive training (curriculum)

Five stages move only the obstacle along world Y: `easy` (1.5 m, clear direct
path), `medium` (0.75 m), `intermediate1` (0.50 m), `intermediate2` (0.25 m),
and `hard` (0 m, original scene).
Robot and target positions, observations, actions, and rewards are unchanged.
Each stage fixes the obstacle position while robot and target spawns vary.
The stage names no longer guarantee increasing difficulty for every spawn. The default is `hard` for compatibility.

Start a new easy policy rather than reusing the unsuccessful hard policy:

```bash
python train.py --stage easy --steps 100000 --output runs/curriculum-easy
python watch_agent.py runs/curriculum-easy/best_model.zip --headless --episodes 1
```

Only proceed once evaluation reports success. If it fails, inspect the behavior
and logs; the suggested step budgets are not guarantees. Continue the learned
policy at the next stage, with a new output directory:

```bash
python train.py --stage medium --resume runs/curriculum-easy/best_model.zip --steps 150000 --output runs/curriculum-medium
python watch_agent.py runs/curriculum-medium/best_model.zip --headless --episodes 1
```

After success at medium, introduce the two intermediate stages. Evaluate each
one and proceed only after success, resuming from the checkpoint that succeeded:

```bash
python train.py --stage intermediate1 --resume runs/curriculum-medium/best_model.zip --steps 150000 --output runs/curriculum-intermediate1
python watch_agent.py runs/curriculum-intermediate1/best_model.zip --headless --episodes 1
```

```bash
python train.py --stage intermediate2 --resume runs/curriculum-intermediate1/best_model.zip --steps 150000 --output runs/curriculum-intermediate2
python watch_agent.py runs/curriculum-intermediate2/best_model.zip --headless --episodes 1
```

Finally return to the original scene, using a new directory to preserve the
earlier hard run:

```bash
python train.py --stage hard --resume runs/curriculum-intermediate2/best_model.zip --steps 300000 --output runs/curriculum-hard-v2
python watch_agent.py runs/curriculum-hard-v2/best_model.zip --headless --episodes 1
```

`--resume` restores network weights and optimizer state; `--steps` is the number
of additional steps, rounded up to complete PPO rollouts. Each run evaluates on
its selected stage and saves its own best model. Stage metadata is stored inside
the model, so `watch_agent.py` uses the correct scene automatically. Older models
without this metadata default to hard. Use `--stage hard` in evaluation to test
any checkpoint on the original scene explicitly. Omit `--headless` to watch;
add `--render` to training to see experience collection.

New policies use `gamma=0.999`. Resuming normally preserves the discount factor
stored in the checkpoint, even if the default in the source code has changed.
Use `--gamma 0.999` to explicitly override it for a resumed experiment; the
effective value is printed at startup and recorded in `config.json`:

```bash
python train.py --stage intermediate1 --resume runs/curriculum-medium/best_model.zip --gamma 0.999 --steps 150000 --output runs/intermediate1-gamma0999-new
```
