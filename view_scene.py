"""Preview the static MuJoCo scene and the head-mounted ray sensor."""

from pathlib import Path
import time

import mujoco
import numpy as np

from raycast import Raycaster


NRAYS = 9
FIELD_OF_VIEW = 120
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

    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("scene.xml")))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot_id = model.body("robot").id
    indicator_id = model.body("robot_indicator").id
    sensor = Raycaster(model, data, {robot_id: indicator_id})
    sensor.setup_raycast(robot_id, NRAYS, FIELD_OF_VIEW)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("top").id
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        while viewer.is_running():
            geomids, distances = sensor.perform_raycast(robot_id, RAY_LENGTH)
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_rays(viewer.user_scn, sensor, robot_id, geomids, distances)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
