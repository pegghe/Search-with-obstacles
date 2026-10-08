"""Behavior checks for rewards and episode endings; no viewer required."""
import unittest

from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np

from raycast_env import RaycastEnv


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = RaycastEnv()
        self.env.reset(seed=42)
        self.env.model.geom_pos[self.env.blocks, :2] = [(0, -1.4), (0, 1.4)]
        self.addCleanup(self.env.close)

    def pose(self, x, y, yaw=0):
        # Joint translations are offsets from the initial robot position (-2, 0).
        self.env.data.qpos[:] = [x + 2, y, yaw]
        self.env.data.qvel[:] = 0
        mujoco.mj_forward(self.env.model, self.env.data)

    def test_random_spawns(self):
        for stage in RaycastEnv.STAGES:
            env = RaycastEnv(stage=stage)
            try:
                positions = []
                for seed in range(50):
                    obs, info = env.reset(seed=seed)
                    robot = env.data.xpos[env.robot_id, :2].copy()
                    target = env.data.geom_xpos[env.target_id, :2].copy()
                    self.assertFalse(env._has_collision())
                    self.assertGreaterEqual(info['target_distance'], env.min_spawn_distance)
                    self.assertTrue(np.all(np.abs(robot) < 3 - env.spawn_clearance))
                    self.assertTrue(np.all(np.abs(target) < 3 - env.spawn_clearance))
                    # A robot placed at the target must also have room to stand.
                    env.data.qpos[env.position_indices] = target - env.model.body_pos[env.robot_id, :2]
                    mujoco.mj_forward(env.model, env.data)
                    self.assertFalse(env._has_collision())
                    repeated, _ = env.reset(seed=seed)
                    np.testing.assert_array_equal(obs, repeated)
                    np.testing.assert_array_equal(robot, env.data.xpos[env.robot_id, :2])
                    np.testing.assert_array_equal(target, env.data.geom_xpos[env.target_id, :2])
                    positions.append(np.concatenate((robot, target)))
                self.assertEqual(len(np.unique(positions, axis=0)), 50)
                env.reset()
                self.assertFalse(np.array_equal(positions[-1], np.concatenate((
                    env.data.xpos[env.robot_id, :2], env.data.geom_xpos[env.target_id, :2]))))
            finally:
                env.close()

    def test_random_obstacle_sizes_and_passages(self):
        samples = []
        positions = []
        for seed in range(50):
            self.env.reset(seed=seed)
            sizes = self.env.model.geom_size[self.env.blocks].copy()
            centers = self.env.model.geom_pos[self.env.blocks].copy()
            positions.append(centers[:, :2])
            samples.append(sizes[:, :2])
            self.assertTrue(np.all(3 - np.abs(centers[:, :2]) - sizes[:, :2] >= 1 - 1e-12))
            gap = np.maximum(np.abs(centers[0, :2] - centers[1, :2]) - sizes[0, :2] - sizes[1, :2], 0)
            self.assertGreaterEqual(np.linalg.norm(gap), 1)
            self.assertTrue(np.all(sizes[:, :2] >= self.env.OBSTACLE_SIZE_MIN))
            self.assertTrue(np.all(sizes[:, :2] <= self.env.OBSTACLE_SIZE_MAX))
            np.testing.assert_array_equal(sizes[:, 2], [0.25, 0.25])
            self.env.reset(seed=seed)
            np.testing.assert_array_equal(sizes, self.env.model.geom_size[self.env.blocks])
            np.testing.assert_array_equal(centers, self.env.model.geom_pos[self.env.blocks])
        self.assertEqual(len(np.unique(np.array(samples).reshape(50, -1), axis=0)), 50)
        self.assertEqual(len(np.unique(np.array(positions).reshape(50, -1), axis=0)), 50)
        # At maximum size, traverse all three horizontal passages, turning at
        # each sampled position. This checks the entire robot, including head.
        self.env.model.geom_size[self.env.blocks, :2] = self.env.OBSTACLE_SIZE_MAX
        self.env.model.geom_pos[self.env.blocks, :2] = [(0, -1.4), (0, 1.4)]
        for y in (-2.5, 0, 2.5):
            for x in np.linspace(-2.5, 2.5, 21):
                for yaw in np.linspace(-np.pi, np.pi, 17):
                    self.pose(x, y, yaw)
                    self.assertFalse(self.env._has_collision(), (x, y, yaw))
        # Enlarged geometry must also be sensed and generate contacts outside
        # the original XML bounds.
        self.pose(-2, -1.4)
        rays = self.env._get_observation().reshape(9, 3)
        self.assertAlmostEqual(float(rays[4, 0]), (2 - 0.43 - 0.60) / 6, places=5)
        self.pose(-0.99, -1.4)
        self.assertTrue(self.env._has_collision())
        # Move the obstacle well outside its original XML position: ray and
        # contact broad-phase bounds must still include it.
        self.env.model.geom_pos[self.env.blocks[0], :2] = [1.2, 0]
        self.pose(-1.5, 0)
        rays = self.env._get_observation().reshape(9, 3)
        self.assertAlmostEqual(float(rays[4, 0]), (2.7 - 0.43 - 0.60) / 6, places=5)
        self.pose(0.81, 0)
        self.assertTrue(self.env._has_collision())

    def test_ray_object_types_and_visibility(self):
        # Target directly in front, clear of the obstacle.
        self.env.model.geom_pos[self.env.target_id, :2] = [0, 2.4]
        self.pose(-2, 2.4)
        rays = self.env._get_observation().reshape(9, 3)
        np.testing.assert_array_equal(rays[4, 1:], [0, 1])
        self.assertAlmostEqual(float(rays[4, 0]), 1.35 / self.env.ray_length, places=5)
        self.assertTrue(np.all(rays[:, 1:].sum(axis=1) <= 1))
        # Target hidden by the lower obstacle.
        self.fixed_scene()
        rays = self.env._get_observation().reshape(9, 3)
        np.testing.assert_array_equal(rays[4, 1:], [1, 0])
        self.assertFalse(rays[:, 2].any())
        # Clear ray toward the east wall.
        self.env.model.geom_pos[self.env.target_id, :2] = [-2, -2]
        self.pose(1, 2.4)
        rays = self.env._get_observation().reshape(9, 3)
        np.testing.assert_array_equal(rays[4, 1:], [1, 0])
        # Target beyond the east wall, then behind the robot.
        for target in ([5, 2.4], [-2.5, 2.4]):
            self.env.model.geom_pos[self.env.target_id, :2] = target
            self.pose(-2, 2.4)
            rays = self.env._get_observation().reshape(9, 3)
            self.assertEqual(rays[4, 2], 0)
            self.assertFalse(rays[:, 2].any())

    def test_target_is_passable(self):
        self.env.model.geom_pos[self.env.target_id, :2] = [0, 2.4]
        self.pose(-0.5, 2.4)
        for _ in range(100):
            _, _, terminated, truncated, info = self.env.step([1, 0])
            self.assertFalse(info['collision'])
            self.assertFalse(any(
                self.env.target_id in (contact.geom1, contact.geom2)
                for contact in self.env.data.contact[:self.env.data.ncon]
            ))
            if terminated or truncated:
                break
        self.assertTrue(info['is_success'])

    def test_gym_api(self):
        check_env(self.env, skip_render_check=True)

    def fixed_scene(self):
        self.env.model.geom_pos[self.env.blocks, :2] = [(0, -1.4), (0, 1.4)]
        self.env.model.geom_pos[self.env.target_id, :2] = [2, -1.4]
        self.pose(-2, -1.4)

    def test_progress_and_step_penalty(self):
        self.fixed_scene()
        _, reward, term, trunc, info = self.env.step([0, 0])
        self.assertAlmostEqual(reward, -0.01)
        self.assertFalse(term or trunc)
        for _ in range(10):
            obs, reward, _, _, info = self.env.step([1, 0])
        self.assertGreater(reward, 0)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertAlmostEqual(reward, sum(info['reward_components'].values()))
        self.env.reset()
        self.fixed_scene()
        _, reward, _, _, info = self.env.step([-1, 0])
        self.assertLess(info['reward_components']['progress'], 0)

    def test_step_penalty_independent_of_duration(self):
        for frame_skip in (1, 4, 8):
            with self.subTest(frame_skip=frame_skip):
                self.env.reset()
                self.env.frame_skip = frame_skip
                for _ in range(3):
                    _, reward, _, _, info = self.env.step([0, 0])
                    self.assertAlmostEqual(reward, -0.01)
                    self.assertEqual(info['reward_components']['step'], -0.01)
                    self.assertNotIn('time', info['reward_components'])

    def test_success_and_reset(self):
        self.fixed_scene()
        self.pose(1.9, -1.4)
        _, reward, term, trunc, info = self.env.step([0, 0])
        self.assertTrue(term and info['is_success'])
        self.assertEqual(info['reward_components']['step'], -0.01)
        self.assertFalse(trunc)
        self.assertGreater(reward, 19)
        obs, info = self.env.reset()
        self.assertEqual(self.env.elapsed_steps, 0)
        self.assertFalse(info['is_success'])
        self.assertEqual(obs.shape, (27,))
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_head_and_wall_collision(self):
        for x, y in [(None, -1.4), (-2, 2.76)]:
            with self.subTest(x=x, y=y):
                self.env.reset()
                self.env.model.geom_pos[self.env.blocks, :2] = [(0, -1.4), (0, 1.4)]
                if x is None:
                    x = -self.env.model.geom_size[self.env.blocks[0], 0] - 0.40
                self.pose(x, y)
                _, reward, term, trunc, info = self.env.step([0, 0])
                self.assertTrue(term and info['collision'])
                self.assertFalse(trunc or info['is_success'])
                self.assertLess(reward, -19)
                self.assertEqual(info['termination_reason'], 'collision')

    def test_drive_into_obstacle(self):
        self.fixed_scene()
        for _ in range(200):
            _, _, term, trunc, info = self.env.step([1, 0])
            if term or trunc:
                break
        self.assertTrue(info['collision'])
        self.assertLess(self.env.elapsed_steps, 200)

    def test_timeout(self):
        for _ in range(self.env.max_episode_steps):
            _, _, term, trunc, info = self.env.step([0, 0])
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info['termination_reason'], 'timeout')
        self.assertAlmostEqual(self.env.data.time, 40)

    def test_collision_priority(self):
        x = -self.env.model.geom_size[self.env.blocks[0], 0] - 0.40
        self.pose(x, -1.4)
        self.env.model.geom_pos[self.env.target_id, :2] = [x, -1.4]
        mujoco.mj_forward(self.env.model, self.env.data)
        _, _, term, _, info = self.env.step([0, 0])
        self.assertTrue(term and info['collision'])
        self.assertFalse(info['is_success'])


if __name__ == '__main__':
    unittest.main()
