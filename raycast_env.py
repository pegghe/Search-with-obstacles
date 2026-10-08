"""Gymnasium navigation environment with sensing, rewards, and episode endings."""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from movement import RobotController
from raycast import Raycaster


class RaycastEnv(gym.Env):
    """Reach the target without touching obstacles; runs without a viewer."""

    metadata = {"render_modes": []}
    STAGES = {"hard": 0.0}
    # Box half-extents in XY; full dimensions are twice these values.
    OBSTACLE_SIZE_MIN = (0.20, 0.30)
    OBSTACLE_SIZE_MAX = (0.60, 0.60)
    MIN_PASSAGE = 1.0

    def __init__(self, stage="hard"):
        super().__init__()
        self.stage = stage
        spec = mujoco.MjSpec.from_file(str(Path(__file__).with_name("scene.xml")))
        # Compile conservative collision/raycast bounds covering the arena.
        # Changing geom_size later does not rebuild MuJoCo's static BVH bounds.
        for name in ("obstacle_1", "obstacle_2"):
            spec.geom(name).pos[:2] = (0, 0)
            spec.geom(name).size[:2] = (3, 3)
        self.model = spec.compile()
        if stage not in self.STAGES:
            raise ValueError(f"Unsupported stage: {stage}; this scene uses two random obstacles")
        self.blocks = [self.model.geom(name).id for name in ("obstacle_1", "obstacle_2")]
        self.model.geom_size[self.blocks, :2] = self.OBSTACLE_SIZE_MAX
        self.data = mujoco.MjData(self.model)
        self.controller = RobotController(self.model, self.data)
        self.robot_id = self.model.body("robot").id
        self.target_id = self.model.geom("target").id
        indicator_id = self.model.body("robot_indicator").id
        self.nrays = 9
        self.ray_length = 6.0
        self.frame_skip = 4  # One action lasts 4 * 0.005 = 0.02 simulated seconds.
        self.max_episode_steps = 2000  # 40 simulated seconds.
        self.target_radius = 0.9  # Robot center must enter the green target disk.
        self.spawn_clearance = 0.47
        self.min_spawn_distance = 1.0
        self.position_indices = [
            int(self.model.joint(name).qposadr[0]) for name in ("robot_x", "robot_y")
        ]
        self.progress_scale = 5.0
        self.step_penalty = -0.01
        self.success_reward = 20.0
        self.collision_penalty = -20.0
        self.robot_geoms = {self.model.geom(name).id for name in ("robot_base", "robot_head")}
        self.obstacle_geoms = {
            self.model.geom(name).id
            for name in ("obstacle_1", "obstacle_2", "wall_north", "wall_south", "wall_east", "wall_west")
        }
        self.elapsed_steps = 0
        self.sensor = Raycaster(self.model, self.data, {self.robot_id: indicator_id})
        self.sensor.setup_raycast(self.robot_id, self.nrays, 120)
        # Action: [forward (avanzamento), turn (rotazione)], both normalized to [-1, 1].
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Per ray: [distance, obstacle, target]; walls count as obstacles.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.nrays * 3,), dtype=np.float32,
        )

    def _get_observation(self):
        """Return only ray measurements, flattened in ray order."""
        geomids, distances = self.sensor.perform_raycast(self.robot_id, self.ray_length)
        ranges = np.where(distances < 0, self.ray_length, distances) / self.ray_length
        observation = np.zeros((self.nrays, 3), dtype=np.float32)
        observation[:, 0] = ranges
        observation[:, 1] = np.isin(geomids, list(self.obstacle_geoms))
        observation[:, 2] = geomids == self.target_id
        return observation.ravel()

    def _target_distance(self):
        delta = self.data.geom_xpos[self.target_id, :2] - self.data.xpos[self.robot_id, :2]
        return float(np.linalg.norm(delta))

    def _has_collision(self):
        """Detect actual robot-obstacle contacts, excluding the floor and sites."""
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist <= 0 and (
                (a in self.robot_geoms and b in self.obstacle_geoms)
                or (b in self.robot_geoms and a in self.obstacle_geoms)
            ):
                return True
        return False

    def _sample_spawn(self):
        """Sample free XY space with room for the robot at either endpoint."""
        margin = self.spawn_clearance
        low = np.array([
            self.model.geom("wall_west").pos[0] + self.model.geom("wall_west").size[0],
            self.model.geom("wall_south").pos[1] + self.model.geom("wall_south").size[1],
        ]) + margin
        high = np.array([
            self.model.geom("wall_east").pos[0] - self.model.geom("wall_east").size[0],
            self.model.geom("wall_north").pos[1] - self.model.geom("wall_north").size[1],
        ]) - margin
        centers = self.model.geom_pos[self.blocks, :2]
        sizes = self.model.geom_size[self.blocks, :2]
        for _ in range(1000):
            position = self.np_random.uniform(low, high)
            delta = np.maximum(np.abs(position - centers) - sizes, 0)
            if np.all(np.linalg.norm(delta, axis=1) > margin):
                return position
        raise RuntimeError("Could not find a free spawn position")

    def _randomize_obstacles(self):
        """Keep disjoint boxes at least 1 m apart and from every arena wall."""
        sizes = self.np_random.uniform(
            self.OBSTACLE_SIZE_MIN, self.OBSTACLE_SIZE_MAX, size=(len(self.blocks), 2),
        )
        low = np.array([
            self.model.geom("wall_west").pos[0] + self.model.geom("wall_west").size[0],
            self.model.geom("wall_south").pos[1] + self.model.geom("wall_south").size[1],
        ])
        high = np.array([
            self.model.geom("wall_east").pos[0] - self.model.geom("wall_east").size[0],
            self.model.geom("wall_north").pos[1] - self.model.geom("wall_north").size[1],
        ])
        for _ in range(1000):
            centers = self.np_random.uniform(
                low + sizes + self.MIN_PASSAGE, high - sizes - self.MIN_PASSAGE,
            )
            gap = np.maximum(np.abs(centers[0] - centers[1]) - sizes[0] - sizes[1], 0)
            if np.linalg.norm(gap) >= self.MIN_PASSAGE:
                self.model.geom_size[self.blocks, :2] = sizes
                self.model.geom_pos[self.blocks, :2] = centers
                return
        raise RuntimeError("Could not place obstacles with enough passage clearance")

    def reset(self, *, seed=None, options=None):
        """Sample robot and target positions and return (observation, info)."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_obstacles()
        robot_position = self._sample_spawn()
        for _ in range(1000):
            target_position = self._sample_spawn()
            if np.linalg.norm(target_position - robot_position) >= self.min_spawn_distance:
                break
        else:
            raise RuntimeError("Could not find sufficiently separated spawn positions")
        # Slide joint positions are offsets from the XML body's initial position.
        self.data.qpos[self.position_indices] = robot_position - self.model.body_pos[self.robot_id, :2]
        self.model.geom_pos[self.target_id, :2] = target_position
        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps = 0
        info = {"target_distance": self._target_distance()}
        info.update(is_success=False, collision=False, termination_reason=None)
        return self._get_observation(), info

    def step(self, action):
        """Apply one action and return the five values expected by Gymnasium."""
        forward, turn = action
        previous_distance = self._target_distance()
        collision = False
        success = False
        for _ in range(self.frame_skip):
            # Update movement and contacts at every physics substep.
            self.controller.apply(forward, turn)
            mujoco.mj_step(self.model, self.data)
            collision = collision or self._has_collision()
            mujoco.mj_forward(self.model, self.data)
            collision = collision or self._has_collision()
            success = not collision and self._target_distance() <= self.target_radius
            if collision or success:
                break
        self.elapsed_steps += 1
        observation = self._get_observation()
        info = {"target_distance": self._target_distance()}
        components = {
            "progress": self.progress_scale * (previous_distance - info["target_distance"]),
            "step": self.step_penalty,
            "success": self.success_reward if success else 0.0,
            "collision": self.collision_penalty if collision else 0.0,
        }
        terminated = collision or success
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        reason = None
        if collision:
            reason = "collision"
        elif success:
            reason = "success"
        elif truncated:
            reason = "timeout"
        info.update(is_success=success, collision=collision, termination_reason=reason,
                    reward_components=components)
        return observation, float(sum(components.values())), terminated, truncated, info
