"""Permanent low-level swerve drive executor plugin.

This plugin owns drive motor actuation and arbitrates drive command sources:
  1) /wall_climber/cmd_vel_manual (temporary keyboard/manual override)
  2) /cmd_vel                    (existing web UI)
  3) /wall_climber/cmd_vel_auto  (autonomous correction)

Twist convention (unchanged):
  linear.x -> left/right on wall
  linear.y -> up/down on wall
  angular.z -> robot yaw
"""

import math

import rclpy
from geometry_msgs.msg import Twist


class SwerveDrivePlugin:
    _SAFE_MIN_STEER_ANGLE = -3.139
    _SAFE_MAX_STEER_ANGLE = 3.139

    def init(self, webots_node, properties):
        self._robot = webots_node.robot
        timestep = int(self._robot.getBasicTimeStep())

        prefixes = ['front_left', 'front_right', 'rear_left', 'rear_right']
        self._steer = []
        self._wheel = []
        self._steer_sensor = []

        for p in prefixes:
            s = self._robot.getDevice(f'{p}_steering_joint')
            w = self._robot.getDevice(f'{p}_wheel_joint')

            if s is not None:
                s.setPosition(0.0)
            else:
                print(f'[SwerveDrive] WARNING: "{p}_steering_joint" not found')
            self._steer.append(s)

            if w is not None:
                w.setPosition(float('inf'))
                w.setVelocity(0.0)
            else:
                print(f'[SwerveDrive] WARNING: "{p}_wheel_joint" not found')
            self._wheel.append(w)

            ps = self._robot.getDevice(f'{p}_steering_joint_sensor')
            if ps is not None:
                ps.enable(timestep)
            self._steer_sensor.append(ps)

        self._drive_speed = float(properties.get('drive_speed', '150.0'))
        self._rotate_speed = float(properties.get('rotate_speed', '60.0'))
        self._steer_threshold = float(properties.get('steer_threshold', '0.03'))
        self._half_length = float(properties.get('half_length', '0.08'))
        self._half_width = float(properties.get('half_width', '0.06'))

        self._manual_timeout_steps = int(properties.get('manual_timeout_steps', '15'))
        self._web_timeout_steps = int(properties.get('web_timeout_steps', '10'))
        self._auto_timeout_steps = int(properties.get('auto_timeout_steps', '15'))

        self._module_positions = [
            (self._half_length, self._half_width),
            (self._half_length, -self._half_width),
            (-self._half_length, self._half_width),
            (-self._half_length, -self._half_width),
        ]
        self._module_radius = max(
            math.hypot(self._half_length, self._half_width),
            1e-6,
        )

        self._manual_cmd = None
        self._web_cmd = None
        self._auto_cmd = None

        self._manual_age = 10**9
        self._web_age = 10**9
        self._auto_age = 10**9

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node('swerve_drive_plugin')

        self._node.create_subscription(
            Twist, '/wall_climber/cmd_vel_manual', self._manual_cb, 1
        )
        self._node.create_subscription(Twist, '/cmd_vel', self._web_cb, 1)
        self._node.create_subscription(
            Twist, '/wall_climber/cmd_vel_auto', self._auto_cb, 1
        )

        n_s = sum(1 for m in self._steer if m)
        n_w = sum(1 for m in self._wheel if m)
        print(
            f'[SwerveDrive] Ready -- {n_s} steer, {n_w} wheel; '
            'prio: manual > web > auto > stop'
        )

    def _manual_cb(self, msg):
        self._manual_cmd = msg
        self._manual_age = 0

    def _web_cb(self, msg):
        self._web_cmd = msg
        self._web_age = 0

    def _auto_cb(self, msg):
        self._auto_cmd = msg
        self._auto_age = 0

    def _increment_ages(self):
        if self._manual_age < 10**9:
            self._manual_age += 1
        if self._web_age < 10**9:
            self._web_age += 1
        if self._auto_age < 10**9:
            self._auto_age += 1

    def _select_cmd(self):
        if (
            self._manual_cmd is not None
            and self._manual_age <= self._manual_timeout_steps
        ):
            return self._manual_cmd, 'manual'
        if self._web_cmd is not None and self._web_age <= self._web_timeout_steps:
            return self._web_cmd, 'web'
        if self._auto_cmd is not None and self._auto_age <= self._auto_timeout_steps:
            return self._auto_cmd, 'auto'
        return None, 'none'

    def _stop_wheels(self):
        for m in self._wheel:
            if m is not None:
                m.setVelocity(0.0)

    def _wrap_pi(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _sanitize_module_command(self, desired_angle, desired_speed, current_angle):
        angle = self._wrap_pi(desired_angle)
        speed = desired_speed

        if current_angle is not None:
            delta = self._wrap_pi(angle - current_angle)
            if abs(delta) > (math.pi / 2.0):
                angle = self._wrap_pi(angle + math.pi)
                speed = -speed

        angle = max(self._SAFE_MIN_STEER_ANGLE, min(self._SAFE_MAX_STEER_ANGLE, angle))
        return angle, speed

    def _compute_swerve_targets(self, cmd):
        lx = float(cmd.linear.x)
        ly = float(cmd.linear.y)
        az = float(cmd.angular.z)

        if abs(lx) < 1e-4 and abs(ly) < 1e-4 and abs(az) < 1e-4:
            return None

        vx = lx * self._drive_speed
        vy = ly * self._drive_speed
        omega = az * (self._rotate_speed / self._module_radius)

        angles = []
        speeds = []
        max_speed = 0.0

        for x_i, y_i in self._module_positions:
            v_ix = vx - omega * y_i
            v_iy = vy + omega * x_i
            angle = math.atan2(v_iy, v_ix)
            speed = math.hypot(v_ix, v_iy)
            angles.append(angle)
            speeds.append(speed)
            if speed > max_speed:
                max_speed = speed

        if max_speed > self._drive_speed and max_speed > 1e-9:
            scale = self._drive_speed / max_speed
            speeds = [s * scale for s in speeds]

        return angles, speeds

    def _apply_targets(self, targets):
        if targets is None:
            self._stop_wheels()
            return

        angles, speeds = targets
        safe_angles = [0.0, 0.0, 0.0, 0.0]
        safe_speeds = [0.0, 0.0, 0.0, 0.0]

        for i in range(4):
            actual = None
            if self._steer_sensor[i] is not None:
                actual = self._steer_sensor[i].getValue()
            safe_angles[i], safe_speeds[i] = self._sanitize_module_command(
                angles[i], speeds[i], actual
            )

        for i, m in enumerate(self._steer):
            if m is not None:
                m.setPosition(safe_angles[i])

        for i, m in enumerate(self._wheel):
            if m is None:
                continue

            v = safe_speeds[i]
            if self._steer_sensor[i] is not None and v != 0.0:
                actual = self._steer_sensor[i].getValue()
                err = abs(self._wrap_pi(safe_angles[i] - actual))
                if err > self._steer_threshold:
                    v = 0.0
            m.setVelocity(v)

    def step(self):
        rclpy.spin_once(self._node, timeout_sec=0)
        self._increment_ages()

        cmd, _source = self._select_cmd()
        targets = self._compute_swerve_targets(cmd) if cmd is not None else None
        self._apply_targets(targets)
