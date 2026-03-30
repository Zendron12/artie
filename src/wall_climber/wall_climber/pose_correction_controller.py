"""Pose-based drift correction controller for horizontal body motion.

This node publishes corrective body motion on /wall_climber/cmd_vel_auto
to hold a target board row and heading while the robot translates
horizontally. It does not manage pen probing or stroke execution.
"""

import math

import rclpy
from geometry_msgs.msg import Pose2D, Twist
from rclpy.node import Node
from std_msgs.msg import String


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PoseCorrectionController(Node):
    def __init__(self):
        super().__init__('pose_correction_controller')

        self.declare_parameter('enabled', False)
        self.declare_parameter('forward_cmd', 0.6)
        self.declare_parameter('target_y', float('nan'))
        self.declare_parameter('target_theta', 0.0)
        self.declare_parameter('auto_capture_y_reference', True)
        self.declare_parameter('auto_capture_theta_reference', False)
        self.declare_parameter('k_y', 1.5)
        self.declare_parameter('k_theta', 2.0)
        self.declare_parameter('omega_sign', 1.0)
        self.declare_parameter('max_lateral_cmd', 0.35)
        self.declare_parameter('max_angular_cmd', 0.35)
        self.declare_parameter('deadband_y', 0.002)
        self.declare_parameter('deadband_theta', 0.01)
        self.declare_parameter('pose_timeout_sec', 0.5)
        self.declare_parameter('publish_zero_on_disable', True)

        self._pose = None
        self._pose_stamp = None
        self._captured_target_y = None
        self._captured_target_theta = None
        self._last_enabled = None
        self._stroke_executor_status = None

        self._last_warn_sec = -1e9
        self._last_debug_sec = -1e9

        self._pose_sub = self.create_subscription(
            Pose2D,
            '/wall_climber/robot_pose_board',
            self._pose_cb,
            10,
        )
        self._stroke_executor_status_sub = self.create_subscription(
            String,
            '/wall_climber/stroke_executor_status',
            self._stroke_executor_status_cb,
            10,
        )
        self._cmd_pub = self.create_publisher(Twist, '/wall_climber/cmd_vel_auto', 10)

        self._timer = self.create_timer(0.05, self._on_timer)  # 20 Hz
        self.get_logger().info('Pose correction controller ready (enabled=false).')

    def _pose_cb(self, msg):
        self._pose = msg
        self._pose_stamp = self.get_clock().now()

    def _stroke_executor_status_cb(self, msg):
        self._stroke_executor_status = str(msg.data).strip()

    def _publish_zero(self):
        msg = Twist()
        self._cmd_pub.publish(msg)

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _log_warn_throttled(self, text, period_sec=2.0):
        now = self._now_sec()
        if now - self._last_warn_sec >= period_sec:
            self.get_logger().warn(text)
            self._last_warn_sec = now

    def _log_debug_throttled(self, text, period_sec=3.0):
        now = self._now_sec()
        if now - self._last_debug_sec >= period_sec:
            self.get_logger().info(text)
            self._last_debug_sec = now

    def _on_timer(self):
        enabled = bool(self.get_parameter('enabled').value)
        forward_cmd = float(self.get_parameter('forward_cmd').value)
        target_y_param = float(self.get_parameter('target_y').value)
        target_theta_param = float(self.get_parameter('target_theta').value)
        auto_capture_y = bool(self.get_parameter('auto_capture_y_reference').value)
        auto_capture_theta = bool(
            self.get_parameter('auto_capture_theta_reference').value
        )
        k_y = float(self.get_parameter('k_y').value)
        k_theta = float(self.get_parameter('k_theta').value)
        omega_sign = float(self.get_parameter('omega_sign').value)
        max_lat = float(self.get_parameter('max_lateral_cmd').value)
        max_ang = float(self.get_parameter('max_angular_cmd').value)
        deadband_y = float(self.get_parameter('deadband_y').value)
        deadband_theta = float(self.get_parameter('deadband_theta').value)
        pose_timeout_sec = float(self.get_parameter('pose_timeout_sec').value)
        publish_zero_on_disable = bool(
            self.get_parameter('publish_zero_on_disable').value
        )

        if self._last_enabled is None:
            self._last_enabled = enabled

        if enabled and not self._last_enabled:
            self._captured_target_y = None
            self._captured_target_theta = None

        if not enabled:
            if self._last_enabled and publish_zero_on_disable:
                self._publish_zero()
            self._last_enabled = enabled
            return

        if self._stroke_executor_status == 'running':
            self._log_warn_throttled(
                'stroke_executor_status=running; suppressing pose correction '
                'output on /wall_climber/cmd_vel_auto.'
            )
            self._last_enabled = enabled
            return

        if self._pose is None or self._pose_stamp is None:
            self._publish_zero()
            self._log_warn_throttled('No robot pose yet; publishing zero /cmd_vel_auto.')
            self._last_enabled = enabled
            return

        age_sec = (
            self.get_clock().now() - self._pose_stamp
        ).nanoseconds * 1e-9
        if age_sec > pose_timeout_sec:
            self._publish_zero()
            self._log_warn_throttled(
                f'Pose timeout ({age_sec:.3f}s > {pose_timeout_sec:.3f}s); '
                'publishing zero /cmd_vel_auto.'
            )
            self._last_enabled = enabled
            return

        current_y = float(self._pose.y)
        current_theta = float(self._pose.theta)

        if math.isnan(target_y_param):
            if auto_capture_y and self._captured_target_y is None:
                self._captured_target_y = current_y
            if auto_capture_y and self._captured_target_y is not None:
                target_y = self._captured_target_y
            else:
                target_y = current_y
        else:
            target_y = target_y_param
            self._captured_target_y = None

        if math.isnan(target_theta_param):
            if auto_capture_theta and self._captured_target_theta is None:
                self._captured_target_theta = current_theta
            if auto_capture_theta and self._captured_target_theta is not None:
                target_theta = self._captured_target_theta
            else:
                target_theta = 0.0
        else:
            target_theta = target_theta_param
            self._captured_target_theta = None

        if target_y is None:
            target_y = current_y
        if target_theta is None:
            target_theta = current_theta

        downward_error = current_y - target_y
        theta_error = _wrap_to_pi(target_theta - current_theta)

        cmd = Twist()
        if abs(theta_error) > 0.20 or abs(downward_error) > 0.05:
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = forward_cmd

        lateral_cmd = _clamp(k_y * downward_error, -max_lat, max_lat)
        if abs(downward_error) < deadband_y:
            lateral_cmd = 0.0
        cmd.linear.y = lateral_cmd

        angular_cmd = _clamp(omega_sign * k_theta * theta_error, -max_ang, max_ang)
        if abs(theta_error) < deadband_theta:
            angular_cmd = 0.0
        cmd.angular.z = angular_cmd

        self._cmd_pub.publish(cmd)

        self._log_debug_throttled(
            '[pose_corr] '
            f'y={current_y:.4f} target_y={target_y:.4f} err_y={downward_error:.4f} | '
            f'theta={current_theta:.4f} target_theta={target_theta:.4f} '
            f'err_theta={theta_error:.4f} | '
            f'cmd=(x={cmd.linear.x:.3f}, y={cmd.linear.y:.3f}, z={cmd.angular.z:.3f})'
        )

        self._last_enabled = enabled


def main(args=None):
    rclpy.init(args=args)
    node = PoseCorrectionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
