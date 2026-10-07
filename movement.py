"""Convert forward and turning commands into planar MuJoCo actuator targets."""

import math


class RobotController:
    """An idealized planar drive, without a wheel or motor model."""

    MAX_SPEED = 0.6  # Meters per second.
    MAX_TURN_RATE = 1.2  # Radians per second, positive counterclockwise.

    def __init__(self, model, data):
        self.data = data
        self.yaw_address = int(model.joint("robot_yaw").qposadr[0])
        self.drive_x = model.actuator("drive_x").id
        self.drive_y = model.actuator("drive_y").id
        self.turn = model.actuator("turn").id

    def apply(self, forward, turn):
        """Apply two normalized commands in [-1, 1] before each physics step."""
        if not math.isfinite(forward) or not math.isfinite(turn):
            raise ValueError("Movement commands must be finite")
        speed = max(-1.0, min(1.0, forward)) * self.MAX_SPEED
        turn_rate = max(-1.0, min(1.0, turn)) * self.MAX_TURN_RATE
        yaw = self.data.qpos[self.yaw_address]

        # Slides use world axes, but forward motion follows the robot's heading.
        # Recompute these targets every step, including while following a curve.
        self.data.ctrl[self.drive_x] = speed * math.cos(yaw)
        self.data.ctrl[self.drive_y] = speed * math.sin(yaw)
        self.data.ctrl[self.turn] = turn_rate
