"""Preview the static MuJoCo scene and the head-mounted ray sensor."""

import time

import mujoco
import numpy as np

from raycast_env import RaycastEnv


RAY_LENGTH = 3.0


def draw_rays(scene, sensor, robot_id, geomids, distances, ray_length=RAY_LENGTH):
    """Append visual ray overlays to a viewer scene without changing physics."""
    origin = sensor.raycast_origins_buf[robot_id]
    directions = sensor.raycast_ray_dirs_buf[robot_id]
    for direction, geom_id, distance in zip(directions, geomids, distances):
        if scene.ngeom >= scene.maxgeom:
            break
        hit = geom_id >= 0 and 0 <= distance <= ray_length
        end = origin + direction * (distance if hit else ray_length)
        color = [1.0, 0.5, 0.0, 1.0] if hit else [0.1, 0.9, 0.2, 1.0]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, origin, end)
        scene.ngeom += 1


def main():
    import mujoco.viewer

    env = RaycastEnv()
    try:
        env.reset(seed=42)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = env.model.camera("top").id
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
            while viewer.is_running():
                geomids, distances = env.sensor.perform_raycast(env.robot_id, env.ray_length)
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_rays(viewer.user_scn, env.sensor, env.robot_id,
                              geomids, distances, ray_length=env.ray_length)
                viewer.sync()
                time.sleep(1 / 60)
    finally:
        env.close()


if __name__ == "__main__":
    main()
