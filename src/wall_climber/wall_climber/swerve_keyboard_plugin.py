"""Temporary keyboard/manual input + pantograph arm controller.

This plugin is intentionally temporary for testing:
- DRIVE: publishes manual Twist overrides to /wall_climber/cmd_vel_manual
- ARM: keeps direct joint control for the pantograph exactly as before

Permanent low-level drive execution lives in swerve_drive_plugin.py.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64


class SwerveKeyboardPlugin:
    # Webots arrow-key codes
    _UP = 315
    _DOWN = 317
    _LEFT = 314
    _RIGHT = 316

    # ── Arm geometry (must match pantograph_arm.xacro) ───────────
    _SH_HALF = 0.05    # half shoulder spacing
    _UPPER_LEN = 0.14  # upper arm length
    _FORE_LEN = 0.18   # forearm length
    _FORE_ANG = 0.281  # fixed forearm offset angle (rad)

    def init(self, webots_node, properties):
        self._robot = webots_node.robot
        timestep = int(self._robot.getBasicTimeStep())

        # ---- keyboard ------------------------------------------------
        self._kb = self._robot.getKeyboard()
        self._kb.enable(timestep)

        # ---- arm motor handles (all 5 joints) -----------------------
        arm_joint_names = [
            'left_shoulder_joint',   # 0
            'left_elbow_joint',      # 1
            'right_shoulder_joint',  # 2
            'right_elbow_joint',     # 3
            'pen_mount_joint',       # 4
        ]
        self._arm_motors = []
        self._arm_sensors = []
        for name in arm_joint_names:
            m = self._robot.getDevice(name)
            if m is None:
                print(f'[SwerveKB] WARNING: "{name}" not found')
            self._arm_motors.append(m)
            s = self._robot.getDevice(f'{name}_sensor')
            if s is not None:
                s.enable(timestep)
            self._arm_sensors.append(s)

        # User-controlled DOFs: left_shoulder, right_shoulder, pen_lift
        self._theta_L = 0.0   # left shoulder angle   (rad)
        self._theta_R = 0.0   # right shoulder angle  (rad)
        self._pen_pos = 0.0   # pen prismatic position (m)

        # ---- tunables ------------------------------------------------
        self._manual_linear_cmd = float(properties.get('manual_linear_cmd', '1.5'))
        self._manual_angular_cmd = float(
            properties.get('manual_angular_cmd', '1.2')
        )

        self._sh_step = float(properties.get('shoulder_step', '0.035'))
        self._pen_step = float(properties.get('pen_step', '0.004'))
        self._shoulder_min = float(properties.get('shoulder_min', '-2.30'))
        self._shoulder_max = float(properties.get('shoulder_max', '2.30'))
        self._elbow_min = float(properties.get('elbow_min', '-2.349'))
        self._elbow_max = float(properties.get('elbow_max', '2.349'))
        self._pen_min_pos = float(properties.get('pen_min_pos', '-0.030'))
        self._pen_max_pos = float(properties.get('pen_max_pos', '0.020'))
        self._pen_gap_deep_limit = float(
            properties.get('pen_gap_deep_limit', '-0.0010')
        )
        self._auto_pen_up_pos = float(properties.get('auto_pen_up_pos', '0.018'))
        self._auto_pen_down_pos = float(
            properties.get('auto_pen_down_pos', '-0.020')
        )
        self._auto_pen_timeout_sec = float(
            properties.get('auto_pen_timeout_sec', '0.5')
        )
        self._stop_pen_on_contact = (
            str(properties.get('stop_pen_on_contact', 'true')).lower() == 'true'
        )
        self._auto_pen_target = self._auto_pen_up_pos
        # Start safely with the pen raised before any controller publishes.
        self._auto_pen_target_time = time.monotonic()
        self._pen_pos = self._auto_pen_up_pos
        self._pen_contact = False
        self._pen_gap = float('nan')
        self._external_theta_L = None
        self._external_theta_R = None
        self._external_arm_target_time = None
        self._external_arm_target_timeout_sec = 0.25

        # ---- ROS 2 manual-drive publisher ----------------------------
        if not rclpy.ok():
            rclpy.init(args=None)
        self._ros_node = rclpy.create_node('swerve_keyboard_driver')
        self._manual_pub = self._ros_node.create_publisher(
            Twist, '/wall_climber/cmd_vel_manual', 1
        )
        self._pen_target_sub = self._ros_node.create_subscription(
            Float64, '/wall_climber/pen_target', self._pen_target_cb, 1
        )
        self._arm_target_sub = self._ros_node.create_subscription(
            JointState,
            '/wall_climber/arm_joint_targets',
            self._arm_target_cb,
            1,
        )
        self._pen_contact_sub = self._ros_node.create_subscription(
            Bool, '/wall_climber/pen_contact', self._pen_contact_cb, 1
        )
        self._pen_gap_sub = self._ros_node.create_subscription(
            Float64, '/wall_climber/pen_gap', self._pen_gap_cb, 1
        )
        self._last_drive_nonzero = False

        n_a = sum(1 for m in self._arm_motors if m)
        print(f'[SwerveKB] Ready -- manual-drive publisher + {n_a}/5 arm motors')
        print('[SwerveKB] DRIVE: W/Up S/Down A/Left D/Right  Q/E=rotate  SPACE=stop')
        print('[SwerveKB] DRIVE: publishes /wall_climber/cmd_vel_manual only')
        print('[SwerveKB] ARM:   1/2=L-shoulder 3/4=R-shoulder 5/6=pen-dn/up H=home')
        print('[SwerveKB] PEN:   auto target from /wall_climber/pen_target when 5/6 idle')
        print('[SwerveKB] PEN:   manual pen position stays latched without fresh auto target')
        print(
            f'[SwerveKB] LIMITS: shoulder=[{self._shoulder_min:.3f},{self._shoulder_max:.3f}] '
            f'elbow=[{self._elbow_min:.3f},{self._elbow_max:.3f}] '
            f'pen=[{self._pen_min_pos:.3f},{self._pen_max_pos:.3f}] '
            f'gap_deep_limit={self._pen_gap_deep_limit:.4f}'
        )
        print(
            '[SwerveKB] ARM:   fresh /wall_climber/arm_joint_targets override '
            f'keyboard shoulders for {self._external_arm_target_timeout_sec:.2f}s'
        )
        print('[SwerveKB] Closed-loop 5-bar constraint active (elbows auto-computed)')

    # ==================================================================
    #  5-bar closed-loop kinematic solver
    # ==================================================================
    def _solve_elbows(self, theta_L, theta_R):
        """Return (phi_L, phi_R) elbow angles that close the loop,
        or None if the configuration is unreachable."""
        SH = self._SH_HALF
        L1 = self._UPPER_LEN
        L2 = self._FORE_LEN
        FA = self._FORE_ANG

        # Shoulder positions (in arm-local XY frame, origin = midpoint)
        s_lx, s_ly = -SH, 0.0
        s_rx, s_ry = SH, 0.0

        # Elbow positions (revolute around Z rotates the +Y direction)
        e_lx = s_lx + L1 * (-math.sin(theta_L))
        e_ly = s_ly + L1 * (math.cos(theta_L))
        e_rx = s_rx + L1 * (-math.sin(theta_R))
        e_ry = s_ry + L1 * (math.cos(theta_R))

        # Distance between elbows
        dx = e_rx - e_lx
        dy = e_ry - e_ly
        d = math.sqrt(dx * dx + dy * dy)

        if d < 1e-9 or d > 2.0 * L2:
            return None  # degenerate or unreachable

        # Circle-circle intersection (both radii = L2)
        a = d / 2.0
        h2 = L2 * L2 - a * a
        if h2 < 0:
            return None
        h = math.sqrt(h2)

        # Midpoint
        mx = (e_lx + e_rx) / 2.0
        my = (e_ly + e_ry) / 2.0

        # Unit vectors: along (E_L→E_R) and perpendicular
        ux, uy = dx / d, dy / d
        # Perpendicular (rotate 90° CCW → points "upward" when arms are symmetric)
        nx, ny = -uy, ux

        # Two solutions — pick the one further in +Y (up the wall)
        p1x, p1y = mx + h * nx, my + h * ny
        p2x, p2y = mx - h * nx, my - h * ny
        if p1y >= p2y:
            px, py = p1x, p1y
        else:
            px, py = p2x, p2y

        # Upper-arm direction angles (from +X axis)
        alpha_L = math.pi / 2.0 + theta_L
        alpha_R = math.pi / 2.0 + theta_R

        # Forearm direction angles (from +X axis)
        beta_L = math.atan2(py - e_ly, px - e_lx)
        beta_R = math.atan2(py - e_ry, px - e_rx)

        # Elbow joint angles (account for fixed forearm offsets)
        #   forearm_total_angle = alpha + phi + offset
        #   left offset = -FA, right offset = +FA
        phi_L = beta_L - alpha_L + FA
        phi_R = beta_R - alpha_R - FA

        # Normalise to [-π, π]
        phi_L = math.atan2(math.sin(phi_L), math.cos(phi_L))
        phi_R = math.atan2(math.sin(phi_R), math.cos(phi_R))

        return phi_L, phi_R

    def _publish_manual_twist(self, lx, ly, az, force=False):
        msg = Twist()
        msg.linear.x = float(lx)
        msg.linear.y = float(ly)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(az)

        nonzero = abs(lx) > 1e-9 or abs(ly) > 1e-9 or abs(az) > 1e-9
        if nonzero:
            self._manual_pub.publish(msg)
            self._last_drive_nonzero = True
            return

        if force or self._last_drive_nonzero:
            self._manual_pub.publish(msg)
            self._last_drive_nonzero = False

    def _pen_target_cb(self, msg):
        self._auto_pen_target = float(msg.data)
        self._auto_pen_target_time = time.monotonic()

    def _pen_contact_cb(self, msg):
        self._pen_contact = bool(msg.data)

    def _pen_gap_cb(self, msg):
        self._pen_gap = float(msg.data)

    def _arm_target_cb(self, msg):
        if len(msg.name) != len(msg.position):
            return
        mapping = dict(zip(msg.name, msg.position))
        if 'left_shoulder_joint' not in mapping or 'right_shoulder_joint' not in mapping:
            return
        self._external_theta_L = float(mapping['left_shoulder_joint'])
        self._external_theta_R = float(mapping['right_shoulder_joint'])
        self._external_arm_target_time = time.monotonic()

    # ------------------------------------------------------------------
    def step(self):
        rclpy.spin_once(self._ros_node, timeout_sec=0)

        keys = set()
        while True:
            k = self._kb.getKey()
            if k == -1:
                break
            keys.add(k)

        # ---- DRIVE (temporary manual input publisher only) -----------
        lin = self._manual_linear_cmd
        ang = self._manual_angular_cmd

        if ord('W') in keys or self._UP in keys:
            self._publish_manual_twist(0.0, lin, 0.0)
        elif ord('S') in keys or self._DOWN in keys:
            self._publish_manual_twist(0.0, -lin, 0.0)
        elif ord('A') in keys or self._LEFT in keys:
            self._publish_manual_twist(-lin, 0.0, 0.0)
        elif ord('D') in keys or self._RIGHT in keys:
            self._publish_manual_twist(lin, 0.0, 0.0)
        elif ord('Q') in keys:
            self._publish_manual_twist(0.0, 0.0, ang)
        elif ord('E') in keys:
            self._publish_manual_twist(0.0, 0.0, -ang)
        elif ord(' ') in keys:
            self._publish_manual_twist(0.0, 0.0, 0.0, force=True)
        else:
            self._publish_manual_twist(0.0, 0.0, 0.0)

        # ---- ARM (closed-loop 5-bar) ---------------------------------
        # Shoulder control: 1/2 = left CW/CCW,  3/4 = right CW/CCW
        if ord('1') in keys:
            self._theta_L += self._sh_step
        if ord('2') in keys:
            self._theta_L -= self._sh_step
        if ord('3') in keys:
            self._theta_R += self._sh_step
        if ord('4') in keys:
            self._theta_R -= self._sh_step
        pen_down_pressed = ord('5') in keys
        pen_up_pressed = ord('6') in keys
        home_pressed = ord('H') in keys

        # Pen: 5=down(-Z toward wall)  6=up(+Z away)
        if pen_down_pressed:
            can_go_deeper = True
            if self._stop_pen_on_contact and self._pen_contact:
                if math.isfinite(self._pen_gap):
                    can_go_deeper = self._pen_gap > self._pen_gap_deep_limit
                else:
                    can_go_deeper = False
            if can_go_deeper:
                self._pen_pos -= self._pen_step
        if pen_up_pressed:
            self._pen_pos += self._pen_step
        # Home
        if home_pressed:
            self._theta_L = 0.0
            self._theta_R = 0.0
            self._pen_pos = self._auto_pen_up_pos

        external_fresh = (
            self._external_arm_target_time is not None
            and (time.monotonic() - self._external_arm_target_time)
            <= self._external_arm_target_timeout_sec
            and self._external_theta_L is not None
            and self._external_theta_R is not None
        )
        if external_fresh:
            self._theta_L = self._external_theta_L
            self._theta_R = self._external_theta_R

        if not pen_down_pressed and not pen_up_pressed and not home_pressed:
            auto_fresh = (
                self._auto_pen_target_time is not None
                and (time.monotonic() - self._auto_pen_target_time) <= self._auto_pen_timeout_sec
            )
            if auto_fresh:
                self._pen_pos = self._auto_pen_target

        # Clamp shoulders
        self._theta_L = max(self._shoulder_min, min(self._shoulder_max, self._theta_L))
        self._theta_R = max(self._shoulder_min, min(self._shoulder_max, self._theta_R))
        self._pen_pos = max(self._pen_min_pos, min(self._pen_max_pos, self._pen_pos))

        # Solve closed-loop constraint for elbows
        result = self._solve_elbows(self._theta_L, self._theta_R)
        if result is not None:
            phi_L, phi_R = result
            phi_L = max(self._elbow_min, min(self._elbow_max, phi_L))
            phi_R = max(self._elbow_min, min(self._elbow_max, phi_R))
            targets = [self._theta_L, phi_L, self._theta_R, phi_R, self._pen_pos]
        else:
            # Unreachable — don't move (hold previous position)
            targets = None

        if targets is not None:
            for i, m in enumerate(self._arm_motors):
                if m is not None:
                    m.setPosition(targets[i])
