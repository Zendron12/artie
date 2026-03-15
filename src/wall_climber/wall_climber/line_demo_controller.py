"""Board-aware two-line writing demo controller (contact-aware, pen-referenced)."""

import json
import math

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D, Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String


IDLE = 'IDLE'
MOVE_TO_LINE1_START = 'MOVE_TO_LINE1_START'
PEN_PROBE_LINE1 = 'PEN_PROBE_LINE1'
PEN_SETTLE_LINE1 = 'PEN_SETTLE_LINE1'
DRAW_LINE1 = 'DRAW_LINE1'
PEN_UP_AFTER_LINE1 = 'PEN_UP_AFTER_LINE1'
MOVE_DOWN = 'MOVE_DOWN'
MOVE_LEFT_TO_LINE2_START = 'MOVE_LEFT_TO_LINE2_START'
PEN_PROBE_LINE2 = 'PEN_PROBE_LINE2'
PEN_SETTLE_LINE2 = 'PEN_SETTLE_LINE2'
DRAW_LINE2 = 'DRAW_LINE2'
PEN_UP_AFTER_LINE2 = 'PEN_UP_AFTER_LINE2'
DONE = 'DONE'


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


class LineDemoController(Node):
    def __init__(self):
        super().__init__('line_demo_controller')

        self.declare_parameter('enabled', False)
        self.declare_parameter('start_margin_x', 0.20)
        self.declare_parameter('end_margin_x', 0.20)
        self.declare_parameter('top_start_margin', 0.12)
        self.declare_parameter('line_spacing', 0.18)

        # Faster body-motion tuning for demo traversal
        self.declare_parameter('draw_speed', 0.45)
        self.declare_parameter('reposition_speed', 0.80)
        self.declare_parameter('target_theta', 0.0)
        self.declare_parameter('k_y', 0.75)
        self.declare_parameter('k_theta', 0.60)
        self.declare_parameter('omega_sign', -1.0)
        self.declare_parameter('max_lateral_cmd', 0.30)
        self.declare_parameter('max_angular_cmd', 0.22)

        self.declare_parameter('pos_tol_x', 0.03)
        self.declare_parameter('pos_tol_y', 0.01)
        self.declare_parameter('theta_tol', 0.03)
        self.declare_parameter('y_capture_clamp_margin', 0.15)

        self.declare_parameter('contact_required_for_drawing', True)
        self.declare_parameter('pen_probe_step', 0.006)
        self.declare_parameter('pen_probe_period_cycles', 1)
        self.declare_parameter('pen_settle_cycles', 4)
        self.declare_parameter('pen_contact_timeout_sec', 1.5)
        self.declare_parameter('pen_pose_timeout_sec', 0.5)
        self.declare_parameter('contact_gap_min', -0.0018)
        self.declare_parameter('contact_gap_max', 0.0018)
        self.declare_parameter('lost_contact_cycles_before_reprobe', 8)
        self.declare_parameter('lost_contact_gap_threshold', 0.004)
        self.declare_parameter('max_probe_retries_per_line', 3)
        self.declare_parameter('draw_pen_extra_depth', 0.006)
        self.declare_parameter('draw_pen_recover_step', 0.0008)

        self.declare_parameter('pen_up_pos', 0.020)
        self.declare_parameter('pen_clear_gap', 0.004)
        self.declare_parameter('pen_lift_timeout_sec', 1.0)
        self.declare_parameter('pen_down_min_pos', -0.010)
        self.declare_parameter('pen_down_max_pos', -0.030)

        self.declare_parameter('publish_zero_on_stop', True)
        self.declare_parameter('pose_timeout_sec', 0.5)

        # Robot body pose (for theta)
        self._pose = None
        self._pose_stamp = None

        # Pen board pose (for x/y alignment and completion)
        self._pen_x = None
        self._pen_y = None
        self._pen_pose_stamp = None

        # Board geometry json
        self._board = None

        # Contact/gap state from supervisor
        self._pen_contact = False
        self._pen_gap = float('nan')
        self._pen_contact_stamp = None
        self._pen_gap_stamp = None

        # FSM
        self._state = IDLE
        self._enabled_last = False

        self._x_start = None
        self._x_end = None
        self._y1 = None
        self._y2 = None
        self._move_down_x_hold = None

        self._probe_target = None
        self._probe_cycle_counter = 0
        self._pen_settle_counter = 0

        self._line1_draw_pen = None
        self._line2_draw_pen = None

        self._line1_lost_contact_cycles = 0
        self._line2_lost_contact_cycles = 0
        self._line1_probe_retries = 0
        self._line2_probe_retries = 0

        self._pen_lift_state_start_sec = None

        self.create_subscription(
            Pose2D, '/wall_climber/robot_pose_board', self._pose_cb, 10
        )
        self.create_subscription(
            PointStamped, '/wall_climber/pen_pose_board', self._pen_pose_cb, 10
        )
        self.create_subscription(
            String, '/wall_climber/board_info', self._board_info_cb, 10
        )
        self.create_subscription(
            Bool, '/wall_climber/pen_contact', self._pen_contact_cb, 10
        )
        self.create_subscription(
            Float64, '/wall_climber/pen_gap', self._pen_gap_cb, 10
        )

        self._cmd_pub = self.create_publisher(Twist, '/wall_climber/cmd_vel_auto', 10)
        self._pen_pub = self.create_publisher(Float64, '/wall_climber/pen_target', 10)

        self._timer = self.create_timer(0.05, self._on_timer)  # 20 Hz
        self.get_logger().info('Line demo controller ready (enabled=false).')

    def _pose_cb(self, msg):
        self._pose = msg
        self._pose_stamp = self.get_clock().now()

    def _pen_pose_cb(self, msg):
        self._pen_x = float(msg.point.x)
        self._pen_y = float(msg.point.y)
        self._pen_pose_stamp = self.get_clock().now()

    def _board_info_cb(self, msg):
        try:
            data = json.loads(msg.data)
            needed = [
                'writable_x_min',
                'writable_x_max',
                'writable_y_min',
                'writable_y_max',
            ]
            if any(k not in data for k in needed):
                self.get_logger().warn('board_info JSON missing writable bounds keys.')
                return
            self._board = data
        except Exception as exc:
            self.get_logger().warn(f'Failed to parse /wall_climber/board_info: {exc}')

    def _pen_contact_cb(self, msg):
        self._pen_contact = bool(msg.data)
        self._pen_contact_stamp = self.get_clock().now()

    def _pen_gap_cb(self, msg):
        self._pen_gap = float(msg.data)
        self._pen_gap_stamp = self.get_clock().now()

    def _publish_pen(self, value):
        m = Float64()
        m.data = float(value)
        self._pen_pub.publish(m)

    def _publish_zero_twist(self):
        self._cmd_pub.publish(Twist())

    def _set_state(self, new_state):
        if self._state == new_state:
            return
        self._state = new_state
        self.get_logger().info(f'entered {new_state}')
        if new_state in (PEN_PROBE_LINE1, PEN_PROBE_LINE2):
            self._probe_cycle_counter = 0
        if new_state in (PEN_SETTLE_LINE1, PEN_SETTLE_LINE2):
            self._pen_settle_counter = 0
        if new_state == DRAW_LINE1:
            self._line1_lost_contact_cycles = 0
        if new_state == DRAW_LINE2:
            self._line2_lost_contact_cycles = 0

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose_fresh(self, timeout_sec):
        if self._pose is None or self._pose_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pose_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_pose_fresh(self, timeout_sec):
        if self._pen_x is None or self._pen_y is None or self._pen_pose_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pen_pose_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_contact_fresh(self, timeout_sec):
        if self._pen_contact_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pen_contact_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_gap_fresh(self, timeout_sec):
        if self._pen_gap_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pen_gap_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_data_fresh(self, timeout_sec):
        return self._pen_contact_fresh(timeout_sec) and self._pen_gap_fresh(timeout_sec)

    def _effective_contact(self, timeout_sec):
        return self._pen_contact_fresh(timeout_sec) and self._pen_contact

    def _gap_contact_good(self, timeout_sec):
        if not self._pen_gap_fresh(timeout_sec):
            return False
        g = self._pen_gap
        g_min = float(self.get_parameter('contact_gap_min').value)
        g_max = float(self.get_parameter('contact_gap_max').value)
        return g_min <= g <= g_max

    def _reset_demo(self):
        self._state = IDLE
        self._x_start = None
        self._x_end = None
        self._y1 = None
        self._y2 = None
        self._move_down_x_hold = None

        self._probe_target = None
        self._probe_cycle_counter = 0
        self._pen_settle_counter = 0

        self._line1_draw_pen = None
        self._line2_draw_pen = None

        self._line1_lost_contact_cycles = 0
        self._line2_lost_contact_cycles = 0
        self._line1_probe_retries = 0
        self._line2_probe_retries = 0

        self._pen_lift_state_start_sec = None

    def _compute_targets(self):
        if self._board is None:
            return False

        start_margin_x = float(self.get_parameter('start_margin_x').value)
        end_margin_x = float(self.get_parameter('end_margin_x').value)
        top_start_margin = float(self.get_parameter('top_start_margin').value)
        line_spacing = float(self.get_parameter('line_spacing').value)

        writable_x_min = float(self._board['writable_x_min'])
        writable_x_max = float(self._board['writable_x_max'])
        writable_y_min = float(self._board['writable_y_min'])
        writable_y_max = float(self._board['writable_y_max'])

        x_start = writable_x_min + start_margin_x
        x_end = writable_x_max - end_margin_x
        if x_end <= x_start:
            self.get_logger().warn('Invalid writable X span after margins; cannot run demo.')
            return False

        y1 = writable_y_min + top_start_margin
        y2 = y1 + line_spacing
        if y2 > writable_y_max:
            y1 = writable_y_max - line_spacing
            y2 = y1 + line_spacing
        if y1 < writable_y_min:
            y1 = writable_y_min
            y2 = min(y1 + line_spacing, writable_y_max)

        self._x_start = x_start
        self._x_end = x_end
        self._y1 = y1
        self._y2 = y2
        return True

    def _tracking_cmd(self, target_x, target_y, target_theta, x_cap, hold_right=False):
        k_y = float(self.get_parameter('k_y').value)
        k_theta = float(self.get_parameter('k_theta').value)
        omega_sign = float(self.get_parameter('omega_sign').value)
        max_lat = float(self.get_parameter('max_lateral_cmd').value)
        max_ang = float(self.get_parameter('max_angular_cmd').value)

        # Pen pose drives positional control.
        x = float(self._pen_x)
        y = float(self._pen_y)
        # Robot body pose still drives heading correction.
        theta = float(self._pose.theta)

        downward_error = y - target_y
        theta_error = _wrap_to_pi(target_theta - theta)

        cmd = Twist()
        cmd.linear.y = _clamp(k_y * downward_error, -max_lat, max_lat)
        cmd.angular.z = _clamp(omega_sign * k_theta * theta_error, -max_ang, max_ang)

        # Keep previous safety gate: reduce x motion when y/theta errors are large.
        if abs(downward_error) > 0.03 or abs(theta_error) > 0.08:
            cmd.linear.x = 0.0
            return cmd

        if hold_right:
            cmd.linear.x = max(0.0, min(float(x_cap), self._x_end - x))
        else:
            x_error = target_x - x
            cmd.linear.x = _clamp(2.0 * x_error, -float(x_cap), float(x_cap))

        return cmd

    def _enter_probe_state(self, state_name, pen_up_pos):
        self._publish_zero_twist()
        self._publish_pen(pen_up_pos)
        self._probe_target = pen_up_pos
        self._probe_cycle_counter = 0
        self._set_state(state_name)

    def _start_pen_up_wait(self, state_name):
        self._pen_lift_state_start_sec = self._now_sec()
        self._set_state(state_name)

    def _handle_pen_up_wait(self, next_state):
        pen_up_pos = float(self.get_parameter('pen_up_pos').value)
        pen_clear_gap = float(self.get_parameter('pen_clear_gap').value)
        pen_lift_timeout_sec = float(self.get_parameter('pen_lift_timeout_sec').value)
        pen_contact_timeout_sec = float(self.get_parameter('pen_contact_timeout_sec').value)

        self._publish_zero_twist()
        self._publish_pen(pen_up_pos)

        lifted = (
            self._pen_data_fresh(pen_contact_timeout_sec)
            and (not self._pen_contact)
            and (self._pen_gap > pen_clear_gap)
        )
        if lifted:
            self._set_state(next_state)
            return

        if self._pen_lift_state_start_sec is None:
            self._pen_lift_state_start_sec = self._now_sec()
            return

        if (self._now_sec() - self._pen_lift_state_start_sec) > pen_lift_timeout_sec:
            self.get_logger().warn(
                f'Pen lift timeout in {self._state}; continuing safely with pen_up command.'
            )
            self._set_state(next_state)

    def _probe_step(self, on_success_state, line_id):
        pen_down_min = float(self.get_parameter('pen_down_min_pos').value)
        pen_down_max = float(self.get_parameter('pen_down_max_pos').value)
        pen_up_pos = float(self.get_parameter('pen_up_pos').value)
        probe_step = float(self.get_parameter('pen_probe_step').value)
        probe_period = max(1, int(self.get_parameter('pen_probe_period_cycles').value))
        contact_timeout_sec = float(self.get_parameter('pen_contact_timeout_sec').value)
        contact_gap_min = float(self.get_parameter('contact_gap_min').value)
        max_retries = max(0, int(self.get_parameter('max_probe_retries_per_line').value))
        draw_pen_extra_depth = float(self.get_parameter('draw_pen_extra_depth').value)

        if self._probe_target is None:
            self._probe_target = pen_up_pos

        if self._gap_contact_good(contact_timeout_sec):
            draw_pen = max(pen_down_max, self._probe_target - draw_pen_extra_depth)
            if on_success_state == DRAW_LINE1:
                self._line1_draw_pen = draw_pen
                self._line1_probe_retries = 0
                self._line1_lost_contact_cycles = 0
                self._set_state(PEN_SETTLE_LINE1)
            else:
                self._line2_draw_pen = draw_pen
                self._line2_probe_retries = 0
                self._line2_lost_contact_cycles = 0
                self._set_state(PEN_SETTLE_LINE2)
            return

        self._publish_zero_twist()

        # Gap-aware probing:
        # - If tip is too deep inside board, lift slightly.
        # - Else keep probing downward gradually.
        if self._pen_gap_fresh(contact_timeout_sec) and self._pen_gap < contact_gap_min:
            self._probe_target = min(pen_up_pos, self._probe_target + probe_step)
            self._probe_cycle_counter = 0
        else:
            # First probing command jumps to min down position, then goes deeper gradually.
            if self._probe_target > pen_down_min:
                self._probe_target = pen_down_min
            else:
                self._probe_cycle_counter += 1
                if self._probe_cycle_counter >= probe_period:
                    self._probe_target = max(pen_down_max, self._probe_target - probe_step)
                    self._probe_cycle_counter = 0

        self._publish_pen(self._probe_target)

        if self._probe_target <= pen_down_max and not self._gap_contact_good(contact_timeout_sec):
            if line_id == 'line1':
                if self._line1_probe_retries < max_retries:
                    self._line1_probe_retries += 1
                    self.get_logger().warn(
                        'Line 1 probe failed, retrying from line 1 start '
                        f'({self._line1_probe_retries}/{max_retries}).'
                    )
                    self._publish_zero_twist()
                    self._publish_pen(pen_up_pos)
                    self._set_state(MOVE_TO_LINE1_START)
                    return
            else:
                if self._line2_probe_retries < max_retries:
                    self._line2_probe_retries += 1
                    self.get_logger().warn(
                        'Line 2 probe failed, retrying from line 2 start '
                        f'({self._line2_probe_retries}/{max_retries}).'
                    )
                    self._publish_zero_twist()
                    self._publish_pen(pen_up_pos)
                    self._set_state(MOVE_LEFT_TO_LINE2_START)
                    return

            self.get_logger().warn(
                f'Pen probe failed after retry budget (line={line_id}, gap={self._pen_gap:.4f} m). '
                'Stopping demo safely.'
            )
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)
            self._set_state(DONE)

    def _update_draw_pen_target(self, target):
        """Keep draw target near calibrated contact gap window during drawing."""
        pen_contact_timeout_sec = float(self.get_parameter('pen_contact_timeout_sec').value)
        contact_gap_min = float(self.get_parameter('contact_gap_min').value)
        contact_gap_max = float(self.get_parameter('contact_gap_max').value)
        recover_step = float(self.get_parameter('draw_pen_recover_step').value)
        pen_down_max = float(self.get_parameter('pen_down_max_pos').value)
        pen_up_pos = float(self.get_parameter('pen_up_pos').value)

        if not self._pen_gap_fresh(pen_contact_timeout_sec):
            return target

        if self._pen_gap > contact_gap_max:
            target -= recover_step
        elif self._pen_gap < contact_gap_min:
            target += recover_step

        return _clamp(target, pen_down_max, pen_up_pos)

    def _on_timer(self):
        enabled = bool(self.get_parameter('enabled').value)
        pen_up_pos = float(self.get_parameter('pen_up_pos').value)
        target_theta = float(self.get_parameter('target_theta').value)
        draw_speed = float(self.get_parameter('draw_speed').value)
        reposition_speed = float(self.get_parameter('reposition_speed').value)
        pos_tol_x = float(self.get_parameter('pos_tol_x').value)
        pos_tol_y = float(self.get_parameter('pos_tol_y').value)
        theta_tol = float(self.get_parameter('theta_tol').value)
        pose_timeout_sec = float(self.get_parameter('pose_timeout_sec').value)
        pen_pose_timeout_sec = float(self.get_parameter('pen_pose_timeout_sec').value)
        pen_contact_timeout_sec = float(self.get_parameter('pen_contact_timeout_sec').value)
        pen_settle_cycles = max(1, int(self.get_parameter('pen_settle_cycles').value))
        lost_contact_gap_threshold = float(
            self.get_parameter('lost_contact_gap_threshold').value
        )
        lost_contact_cycles_before_reprobe = max(
            1, int(self.get_parameter('lost_contact_cycles_before_reprobe').value)
        )
        publish_zero_on_stop = bool(self.get_parameter('publish_zero_on_stop').value)
        contact_required = bool(self.get_parameter('contact_required_for_drawing').value)

        if not enabled:
            if self._enabled_last and publish_zero_on_stop:
                self._publish_zero_twist()
                self._publish_pen(pen_up_pos)
            self._reset_demo()
            self._enabled_last = False
            return

        if enabled and not self._enabled_last:
            self._reset_demo()
            self.get_logger().info('enabled=true, waiting for board/pose then starting demo.')

        if self._board is None or not self._pose_fresh(pose_timeout_sec):
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)
            self._enabled_last = True
            return

        if not self._pen_pose_fresh(pen_pose_timeout_sec):
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)
            self._enabled_last = True
            return

        if self._state == IDLE and self._compute_targets():
            self.get_logger().info(
                f'demo targets: x_start={self._x_start:.3f} x_end={self._x_end:.3f} '
                f'y1={self._y1:.3f} y2={self._y2:.3f}'
            )
            self._set_state(MOVE_TO_LINE1_START)

        pen_x = float(self._pen_x)
        pen_y = float(self._pen_y)
        theta = float(self._pose.theta)
        theta_error = _wrap_to_pi(target_theta - theta)

        if self._state == IDLE:
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)

        elif self._state == MOVE_TO_LINE1_START:
            self._publish_pen(pen_up_pos)
            cmd = self._tracking_cmd(
                self._x_start,
                self._y1,
                target_theta,
                reposition_speed,
                hold_right=False,
            )
            self._cmd_pub.publish(cmd)
            if (
                abs(pen_x - self._x_start) < pos_tol_x
                and abs(pen_y - self._y1) < pos_tol_y
                and abs(theta_error) < theta_tol
            ):
                self._enter_probe_state(PEN_PROBE_LINE1, pen_up_pos)

        elif self._state == PEN_PROBE_LINE1:
            self._probe_step(DRAW_LINE1, 'line1')

        elif self._state == PEN_SETTLE_LINE1:
            self._publish_zero_twist()
            if self._line1_draw_pen is None:
                self._line1_draw_pen = float(self.get_parameter('pen_down_min_pos').value)
            self._publish_pen(self._line1_draw_pen)
            self._pen_settle_counter += 1
            if self._pen_settle_counter >= pen_settle_cycles:
                self._set_state(DRAW_LINE1)

        elif self._state == DRAW_LINE1:
            if contact_required:
                if self._gap_contact_good(pen_contact_timeout_sec):
                    self._line1_lost_contact_cycles = 0
                elif (
                    self._pen_gap_fresh(pen_contact_timeout_sec)
                    and self._pen_gap > lost_contact_gap_threshold
                ):
                    self._line1_lost_contact_cycles += 1
                    if self._line1_lost_contact_cycles >= lost_contact_cycles_before_reprobe:
                        self.get_logger().warn('Contact lost during DRAW_LINE1, re-probing.')
                        self._enter_probe_state(PEN_PROBE_LINE1, pen_up_pos)
                        return
                else:
                    self._line1_lost_contact_cycles = 0

            if self._line1_draw_pen is None:
                self._line1_draw_pen = float(self.get_parameter('pen_down_min_pos').value)
            self._line1_draw_pen = self._update_draw_pen_target(self._line1_draw_pen)
            self._publish_pen(self._line1_draw_pen)
            cmd = self._tracking_cmd(
                self._x_end,
                self._y1,
                target_theta,
                draw_speed,
                hold_right=True,
            )
            self._cmd_pub.publish(cmd)
            if pen_x >= (self._x_end - pos_tol_x):
                self._move_down_x_hold = pen_x
                self._start_pen_up_wait(PEN_UP_AFTER_LINE1)

        elif self._state == PEN_UP_AFTER_LINE1:
            self._handle_pen_up_wait(MOVE_DOWN)

        elif self._state == MOVE_DOWN:
            self._publish_pen(pen_up_pos)
            target_x = pen_x if self._move_down_x_hold is None else self._move_down_x_hold
            cmd = self._tracking_cmd(
                target_x,
                self._y2,
                target_theta,
                reposition_speed,
                hold_right=False,
            )
            self._cmd_pub.publish(cmd)
            if abs(pen_y - self._y2) < pos_tol_y:
                self._set_state(MOVE_LEFT_TO_LINE2_START)

        elif self._state == MOVE_LEFT_TO_LINE2_START:
            self._publish_pen(pen_up_pos)
            cmd = self._tracking_cmd(
                self._x_start,
                self._y2,
                target_theta,
                reposition_speed,
                hold_right=False,
            )
            self._cmd_pub.publish(cmd)
            if (
                abs(pen_x - self._x_start) < pos_tol_x
                and abs(pen_y - self._y2) < pos_tol_y
                and abs(theta_error) < theta_tol
            ):
                self._enter_probe_state(PEN_PROBE_LINE2, pen_up_pos)

        elif self._state == PEN_PROBE_LINE2:
            self._probe_step(DRAW_LINE2, 'line2')

        elif self._state == PEN_SETTLE_LINE2:
            self._publish_zero_twist()
            if self._line2_draw_pen is None:
                self._line2_draw_pen = float(self.get_parameter('pen_down_min_pos').value)
            self._publish_pen(self._line2_draw_pen)
            self._pen_settle_counter += 1
            if self._pen_settle_counter >= pen_settle_cycles:
                self._set_state(DRAW_LINE2)

        elif self._state == DRAW_LINE2:
            if contact_required:
                if self._gap_contact_good(pen_contact_timeout_sec):
                    self._line2_lost_contact_cycles = 0
                elif (
                    self._pen_gap_fresh(pen_contact_timeout_sec)
                    and self._pen_gap > lost_contact_gap_threshold
                ):
                    self._line2_lost_contact_cycles += 1
                    if self._line2_lost_contact_cycles >= lost_contact_cycles_before_reprobe:
                        self.get_logger().warn('Contact lost during DRAW_LINE2, re-probing.')
                        self._enter_probe_state(PEN_PROBE_LINE2, pen_up_pos)
                        return
                else:
                    self._line2_lost_contact_cycles = 0

            if self._line2_draw_pen is None:
                self._line2_draw_pen = float(self.get_parameter('pen_down_min_pos').value)
            self._line2_draw_pen = self._update_draw_pen_target(self._line2_draw_pen)
            self._publish_pen(self._line2_draw_pen)
            cmd = self._tracking_cmd(
                self._x_end,
                self._y2,
                target_theta,
                draw_speed,
                hold_right=True,
            )
            self._cmd_pub.publish(cmd)
            if pen_x >= (self._x_end - pos_tol_x):
                self._start_pen_up_wait(PEN_UP_AFTER_LINE2)

        elif self._state == PEN_UP_AFTER_LINE2:
            self._handle_pen_up_wait(DONE)

        elif self._state == DONE:
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)

        self._enabled_last = True


def main(args=None):
    rclpy.init(args=args)
    node = LineDemoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
