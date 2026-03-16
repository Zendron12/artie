"""Generic board-aware stroke executor for Artie.

This node executes JSON stroke plans in board coordinates using the same
contact-aware drawing behavior proven in line_demo_controller.
"""

import json
import math

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D, Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String


IDLE = 'IDLE'
MOVE_TO_STROKE_START = 'MOVE_TO_STROKE_START'
PEN_PROBE = 'PEN_PROBE'
PEN_SETTLE = 'PEN_SETTLE'
DRAW_SEGMENT = 'DRAW_SEGMENT'
CORNER_SETTLE = 'CORNER_SETTLE'  # <--- تمت إضافة حالة الزاوية هنا
PEN_UP = 'PEN_UP'
ADVANCE_SEGMENT = 'ADVANCE_SEGMENT'
ADVANCE_STROKE = 'ADVANCE_STROKE'
DONE = 'DONE'


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


class StrokeExecutor(Node):
    def __init__(self):
        super().__init__('stroke_executor')

        self.declare_parameter('enabled', False)
        self.declare_parameter('draw_speed', 0.10)
        self.declare_parameter('reposition_speed', 0.80)
        self.declare_parameter('target_theta', 0.0)
        self.declare_parameter('k_y', 0.75)
        self.declare_parameter('k_theta', 0.60)
        self.declare_parameter('omega_sign', -1.0)
        self.declare_parameter('max_lateral_cmd', 0.30)
        self.declare_parameter('max_angular_cmd', 0.22)

        self.declare_parameter('pos_tol_x', 0.004)
        self.declare_parameter('pos_tol_y', 0.004)
        self.declare_parameter('theta_tol', 0.03)

        self.declare_parameter('contact_required_for_drawing', True)
        self.declare_parameter('pen_probe_step', 0.0025)
        self.declare_parameter('pen_probe_period_cycles', 1)
        self.declare_parameter('pen_settle_cycles', 4)
        self.declare_parameter('corner_settle_cycles', 3)
        self.declare_parameter('draw_start_delay_cycles', 0)
        self.declare_parameter('pen_contact_timeout_sec', 1.5)
        self.declare_parameter('pen_pose_timeout_sec', 0.5)
        self.declare_parameter('contact_gap_min', -0.0018)
        self.declare_parameter('contact_gap_max', 0.0018)
        self.declare_parameter('lost_contact_cycles_before_reprobe', 8)
        self.declare_parameter('lost_contact_gap_threshold', 0.004)
        self.declare_parameter('max_probe_retries_per_line', 3)
        self.declare_parameter('draw_pen_extra_depth', 0.0) # 1 ملم ضغط للخط الناعم
        self.declare_parameter('draw_pen_recover_step', 0.0)

        self.declare_parameter('pen_up_pos', 0.018)
        self.declare_parameter('pen_clear_gap', 0.004)
        self.declare_parameter('pen_lift_timeout_sec', 1.5)
        self.declare_parameter('pen_down_min_pos', -0.010)
        self.declare_parameter('pen_down_max_pos', -0.030)

        self.declare_parameter('publish_zero_on_stop', True)
        self.declare_parameter('pose_timeout_sec', 0.5)

        self._pose = None
        self._pose_stamp = None
        self._pen_x = None
        self._pen_y = None
        self._pen_pose_stamp = None
        self._board = None
        self._pen_contact = False
        self._pen_gap = float('nan')
        self._pen_contact_stamp = None
        self._pen_gap_stamp = None

        self._state = IDLE
        self._enabled_last = False
        self._status = None

        self._current_plan = None
        self._pending_plan = None
        self._stroke_index = 0
        self._segment_index = 0

        self._probe_target = None
        self._probe_cycle_counter = 0
        self._pen_settle_counter = 0
        self._corner_settle_counter = 0  # العداد الجديد
        self._draw_pen_target = None
        self._lost_contact_cycles = 0
        self._probe_retries = 0
        self._pen_lift_state_start_sec = None
        self._next_state_after_pen_up = None
        self._draw_segment_cycles = 0

        self.create_subscription(
            String, '/wall_climber/stroke_plan', self._plan_cb, 10
        )
        self.create_subscription(
            Pose2D, '/wall_climber/robot_pose_board', self._pose_cb, 10
        )
        self.create_subscription(
            PointStamped, '/wall_climber/pen_pose_board', self._pen_pose_cb, 10
        )
        self.create_subscription(
            Bool, '/wall_climber/pen_contact', self._pen_contact_cb, 10
        )
        self.create_subscription(
            Float64, '/wall_climber/pen_gap', self._pen_gap_cb, 10
        )
        self.create_subscription(
            String, '/wall_climber/board_info', self._board_info_cb, 10
        )

        self._cmd_pub = self.create_publisher(Twist, '/wall_climber/cmd_vel_auto', 10)
        self._pen_pub = self.create_publisher(Float64, '/wall_climber/pen_target', 10)
        self._status_pub = self.create_publisher(
            String, '/wall_climber/stroke_executor_status', 10
        )
        self._drawing_active_pub = self.create_publisher(
            Bool, '/wall_climber/drawing_active', 10
        )

        self._timer = self.create_timer(0.05, self._on_timer)
        self._set_status('idle')
        self.get_logger().info('Stroke executor ready (enabled=false).')

    def _pose_cb(self, msg):
        self._pose = msg
        self._pose_stamp = self.get_clock().now()

    def _pen_pose_cb(self, msg):
        self._pen_x = float(msg.point.x)
        self._pen_y = float(msg.point.y)
        self._pen_pose_stamp = self.get_clock().now()

    def _pen_contact_cb(self, msg):
        self._pen_contact = bool(msg.data)
        self._pen_contact_stamp = self.get_clock().now()

    def _pen_gap_cb(self, msg):
        self._pen_gap = float(msg.data)
        self._pen_gap_stamp = self.get_clock().now()

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

    def _plan_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self._reject_plan(f'Malformed stroke plan JSON: {exc}')
            return

        normalized, error = self._normalize_plan(payload)
        if error is not None:
            self._reject_plan(error)
            return

        self._pending_plan = normalized
        if self._board is None:
            self.get_logger().warn(
                'Received structurally valid stroke plan, but board_info is not ready yet; '
                'deferring writable-area validation.'
            )
            return

        self._finalize_pending_plan()

    def _normalize_plan(self, payload):
        if not isinstance(payload, dict):
            return None, 'Stroke plan must be a JSON object.'

        frame = payload.get('frame')
        if frame != 'board':
            return None, 'stroke_plan.frame must be exactly "board".'

        strokes = payload.get('strokes')
        if not isinstance(strokes, list):
            return None, 'stroke_plan.strokes must be a list.'

        normalized_strokes = []
        for index, stroke in enumerate(strokes):
            if not isinstance(stroke, dict):
                return None, f'stroke[{index}] must be an object.'

            stroke_type = stroke.get('type')
            if stroke_type not in ('line', 'polyline'):
                return None, f'stroke[{index}].type must be "line" or "polyline".'

            draw = stroke.get('draw')
            if not isinstance(draw, bool):
                return None, f'stroke[{index}].draw must be boolean.'

            points = stroke.get('points')
            if not isinstance(points, list):
                return None, f'stroke[{index}].points must be a list.'

            normalized_points = []
            for point_index, point in enumerate(points):
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    return None, (
                        f'stroke[{index}].points[{point_index}] must be [x, y].'
                    )
                try:
                    x = float(point[0])
                    y = float(point[1])
                except Exception:
                    return None, (
                        f'stroke[{index}].points[{point_index}] must contain numeric values.'
                    )
                if not math.isfinite(x) or not math.isfinite(y):
                    return None, (
                        f'stroke[{index}].points[{point_index}] must contain finite values.'
                    )
                normalized_points.append((x, y))

            if stroke_type == 'line' and len(normalized_points) != 2:
                return None, f'stroke[{index}] of type line must contain exactly 2 points.'
            if stroke_type == 'polyline' and len(normalized_points) < 2:
                return None, (
                    f'stroke[{index}] of type polyline must contain at least 2 points.'
                )

            normalized_strokes.append(
                {
                    'type': stroke_type,
                    'draw': draw,
                    'points': normalized_points,
                }
            )

        return {'frame': 'board', 'strokes': normalized_strokes}, None

    def _segment_count_total(self, plan):
        return sum(max(len(stroke['points']) - 1, 0) for stroke in plan['strokes'])

    def _segment_is_axis_aligned(self, start_point, end_point, epsilon=1e-6):
        dx = abs(float(end_point[0]) - float(start_point[0]))
        dy = abs(float(end_point[1]) - float(start_point[1]))
        return dx <= epsilon or dy <= epsilon

    def _point_inside_writable(self, point):
        if self._board is None:
            return False
        x, y = point
        return (
            float(self._board['writable_x_min']) <= x <= float(self._board['writable_x_max'])
            and float(self._board['writable_y_min']) <= y <= float(self._board['writable_y_max'])
        )

    def _validate_plan_points(self, plan):
        if self._board is None:
            return 'board_info is not available; cannot validate stroke points yet.'

        for stroke_index, stroke in enumerate(plan['strokes']):
            for point_index, point in enumerate(stroke['points']):
                if not self._point_inside_writable(point):
                    return (
                        'Stroke plan rejected: '
                        f'stroke[{stroke_index}].points[{point_index}]={point} '
                        'is outside writable board bounds.'
                    )
            for segment_index in range(len(stroke['points']) - 1):
                start_point = stroke['points'][segment_index]
                end_point = stroke['points'][segment_index + 1]
                if not self._segment_is_axis_aligned(start_point, end_point):
                    return (
                        'Stroke plan rejected: '
                        f'stroke[{stroke_index}] segment[{segment_index}] '
                        f'from {start_point} to {end_point} is not axis-aligned. '
                        'v1 supports only horizontal/vertical line and polyline segments.'
                    )
        return None

    def _finalize_pending_plan(self):
        if self._pending_plan is None:
            return

        error = self._validate_plan_points(self._pending_plan)
        if error is not None:
            self._pending_plan = None
            self._reject_plan(error)
            return

        self._current_plan = self._pending_plan
        self._pending_plan = None
        self._reset_execution()

        stroke_count = len(self._current_plan['strokes'])
        segment_count = self._segment_count_total(self._current_plan)
        first_stroke_type = (
            self._current_plan['strokes'][0]['type'] if stroke_count > 0 else 'none'
        )
        self.get_logger().info(
            f'received plan: {stroke_count} strokes, {segment_count} segments total, '
            f'first stroke={first_stroke_type}'
        )
        if stroke_count == 0:
            self.get_logger().info('Loaded empty stroke plan; staying idle safely.')
            self._set_status('idle')
            self._state = IDLE
            return

        if bool(self.get_parameter('enabled').value):
            self._set_status('running')
            self._set_state(MOVE_TO_STROKE_START)
        else:
            self._set_status('idle')
            self._state = IDLE

    def _reject_plan(self, reason):
        self.get_logger().warn(reason)
        self._publish_zero_twist()
        self._publish_pen(float(self.get_parameter('pen_up_pos').value))
        self._set_status('error')
        self._state = DONE

    def _publish_pen(self, value):
        msg = Float64()
        msg.data = float(value)
        self._pen_pub.publish(msg)

    def _publish_drawing_active(self, active):
        msg = Bool()
        msg.data = bool(active)
        self._drawing_active_pub.publish(msg)

    def _publish_zero_twist(self):
        self._cmd_pub.publish(Twist())

    def _set_status(self, text):
        if self._status == text:
            return
        self._status = text
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)

    def _set_state(self, new_state):
        if self._state == new_state:
            return
        previous_state = self._state
        self._state = new_state
        self.get_logger().info(
            f'entered {new_state} '
            f'(stroke={self._stroke_index}, segment={self._segment_index})'
        )
        if new_state == PEN_PROBE:
            self._probe_cycle_counter = 0
        if new_state == PEN_SETTLE:
            self._pen_settle_counter = 0
        if new_state == CORNER_SETTLE:
            self._corner_settle_counter = 0
        if new_state == DRAW_SEGMENT:
            self._lost_contact_cycles = 0
            if previous_state == PEN_SETTLE:
                self._draw_segment_cycles = 0

    def _reset_execution(self):
        self._state = IDLE
        self._stroke_index = 0
        self._segment_index = 0
        self._probe_target = None
        self._probe_cycle_counter = 0
        self._pen_settle_counter = 0
        self._corner_settle_counter = 0
        self._draw_pen_target = None
        self._lost_contact_cycles = 0
        self._probe_retries = 0
        self._pen_lift_state_start_sec = None
        self._next_state_after_pen_up = None
        self._draw_segment_cycles = 0

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

    def _gap_contact_good(self, timeout_sec):
        if not self._pen_gap_fresh(timeout_sec):
            return False
        gap = self._pen_gap
        gap_min = float(self.get_parameter('contact_gap_min').value)
        gap_max = float(self.get_parameter('contact_gap_max').value)
        return gap_min <= gap <= gap_max

    def _current_stroke(self):
        if self._current_plan is None:
            return None
        strokes = self._current_plan.get('strokes', [])
        if not (0 <= self._stroke_index < len(strokes)):
            return None
        return strokes[self._stroke_index]

    def _current_segment_points(self):
        stroke = self._current_stroke()
        if stroke is None:
            return None, None
        points = stroke['points']
        if not (0 <= self._segment_index < len(points) - 1):
            return None, None
        return points[self._segment_index], points[self._segment_index + 1]

    def _segment_is_last(self):
        stroke = self._current_stroke()
        if stroke is None:
            return True
        return self._segment_index >= (len(stroke['points']) - 2)

    def _tracking_cmd(self, target_x, target_y, target_theta, speed_cap):
        k_y = float(self.get_parameter('k_y').value)
        k_theta = float(self.get_parameter('k_theta').value)
        omega_sign = float(self.get_parameter('omega_sign').value)
        max_lat = float(self.get_parameter('max_lateral_cmd').value)
        max_ang = float(self.get_parameter('max_angular_cmd').value)

        x = float(self._pen_x)
        y = float(self._pen_y)
        theta = float(self._pose.theta)

        x_error = target_x - x
        downward_error = y - target_y
        theta_error = _wrap_to_pi(target_theta - theta)

        cmd = Twist()
        cmd.linear.y = _clamp(k_y * downward_error, -max_lat, max_lat)
        cmd.angular.z = _clamp(omega_sign * k_theta * theta_error, -max_ang, max_ang)

        if abs(downward_error) > 0.03 or abs(theta_error) > 0.08:
            cmd.linear.x = 0.0
            return cmd

        cmd.linear.x = _clamp(2.0 * x_error, -float(speed_cap), float(speed_cap))
        return cmd

    def _segment_complete(self, target_x, target_y, target_theta):
        pos_tol_x = float(self.get_parameter('pos_tol_x').value)
        pos_tol_y = float(self.get_parameter('pos_tol_y').value)
        theta_tol = float(self.get_parameter('theta_tol').value)

        theta_error = _wrap_to_pi(target_theta - float(self._pose.theta))
        return (
            abs(float(self._pen_x) - target_x) < pos_tol_x
            and abs(float(self._pen_y) - target_y) < pos_tol_y
            and abs(theta_error) < theta_tol
        )

    def _enter_probe_state(self, pen_up_pos):
        self._publish_zero_twist()
        self._publish_pen(pen_up_pos)
        self._probe_target = pen_up_pos
        self._probe_cycle_counter = 0
        self._set_state(PEN_PROBE)

    def _start_pen_up_wait(self, next_state):
        self._pen_lift_state_start_sec = self._now_sec()
        self._next_state_after_pen_up = next_state
        self._set_state(PEN_UP)

    def _handle_pen_up_wait(self):
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
            self._set_state(self._next_state_after_pen_up or ADVANCE_STROKE)
            return

        if self._pen_lift_state_start_sec is None:
            self._pen_lift_state_start_sec = self._now_sec()
            return

        if (self._now_sec() - self._pen_lift_state_start_sec) > pen_lift_timeout_sec:
            self.get_logger().warn(
                f'Pen lift timeout in {self._state}; continuing safely with pen_up command.'
            )
            self._set_state(self._next_state_after_pen_up or ADVANCE_STROKE)

    def _probe_step(self):
        pen_down_min = float(self.get_parameter('pen_down_min_pos').value)
        pen_down_max = float(self.get_parameter('pen_down_max_pos').value)
        pen_up_pos = float(self.get_parameter('pen_up_pos').value)
        probe_step = float(self.get_parameter('pen_probe_step').value)
        probe_period = max(1, int(self.get_parameter('pen_probe_period_cycles').value))
        contact_timeout_sec = float(self.get_parameter('pen_contact_timeout_sec').value)
        contact_gap_min = float(self.get_parameter('contact_gap_min').value)
        max_retries = max(0, int(self.get_parameter('max_probe_retries_per_line').value))
        draw_pen_extra_depth = float(self.get_parameter('draw_pen_extra_depth').value)

        if not self._pen_data_fresh(contact_timeout_sec):
            self._fail_execution(
                'Pen contact/gap data stale during PEN_PROBE; stopping safely.'
            )
            return

        if self._probe_target is None:
            self._probe_target = pen_up_pos

        if self._gap_contact_good(contact_timeout_sec):
            self._draw_pen_target = max(
                pen_down_max,
                self._probe_target - draw_pen_extra_depth,
            )
            self._probe_retries = 0
            self._lost_contact_cycles = 0
            self._set_state(PEN_SETTLE)
            return

        self._publish_zero_twist()

        if self._pen_gap_fresh(contact_timeout_sec) and self._pen_gap < contact_gap_min:
            self._probe_target = min(pen_up_pos, self._probe_target + probe_step)
            self._probe_cycle_counter = 0
        else:
            if self._probe_target > pen_down_min:
                self._probe_target = pen_down_min
            else:
                self._probe_cycle_counter += 1
                if self._probe_cycle_counter >= probe_period:
                    self._probe_target = max(pen_down_max, self._probe_target - probe_step)
                    self._probe_cycle_counter = 0

        self._publish_pen(self._probe_target)

        if self._probe_target <= pen_down_max and not self._gap_contact_good(contact_timeout_sec):
            if self._probe_retries < max_retries:
                self._probe_retries += 1
                self.get_logger().warn(
                    'Pen probe failed, retrying from the current segment start '
                    f'({self._probe_retries}/{max_retries}).'
                )
                self._publish_zero_twist()
                self._publish_pen(pen_up_pos)
                self._set_state(MOVE_TO_STROKE_START)
                return

            self._fail_execution(
                'Pen probe failed after retry budget; stopping stroke execution safely.'
            )

    def _update_draw_pen_target(self, target):
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

    def _fail_execution(self, reason):
        self.get_logger().warn(reason)
        self._publish_zero_twist()
        self._publish_pen(float(self.get_parameter('pen_up_pos').value))
        self._set_status('error')
        self._state = DONE

    def _on_timer(self):
        enabled = bool(self.get_parameter('enabled').value)
        pen_up_pos = float(self.get_parameter('pen_up_pos').value)
        target_theta = float(self.get_parameter('target_theta').value)
        draw_speed = float(self.get_parameter('draw_speed').value)
        reposition_speed = float(self.get_parameter('reposition_speed').value)
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

        if self._pending_plan is not None and self._board is not None:
            self._finalize_pending_plan()

        if not enabled:
            if self._enabled_last and publish_zero_on_stop:
                self._publish_zero_twist()
                self._publish_pen(pen_up_pos)
            # Re-enable drawing for manual control when stroke executor is disabled
            self._publish_drawing_active(True)
            self._reset_execution()
            self._set_status('idle')
            self._enabled_last = False
            return

        if enabled and not self._enabled_last:
            self._reset_execution()
            if self._current_plan is not None and len(self._current_plan['strokes']) > 0:
                self._set_status('running')
                self._set_state(MOVE_TO_STROKE_START)
                self.get_logger().info('enabled=true, starting loaded stroke plan.')
            else:
                self.get_logger().info('enabled=true, waiting for a valid stroke plan.')
                self._set_status('idle')

        if self._board is None or not self._pose_fresh(pose_timeout_sec):
            if self._current_plan is not None and self._state not in (IDLE, DONE):
                self._fail_execution(
                    'Robot pose/board data stale during stroke execution; stopping safely.'
                )
            else:
                self._publish_zero_twist()
                self._publish_pen(pen_up_pos)
                self._set_status('idle')
            self._enabled_last = True
            return

        if not self._pen_pose_fresh(pen_pose_timeout_sec):
            if self._current_plan is not None and self._state not in (IDLE, DONE):
                self._fail_execution('Pen pose stale during stroke execution; stopping safely.')
            else:
                self._publish_zero_twist()
                self._publish_pen(pen_up_pos)
                self._set_status('idle')
            self._enabled_last = True
            return

        if self._current_plan is None:
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)
            self._set_status('idle')
            self._enabled_last = True
            return

        if self._state == IDLE:
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)

        elif self._state == MOVE_TO_STROKE_START:
            stroke = self._current_stroke()
            if stroke is None:
                self._set_state(DONE)
            else:
                start_point, _ = self._current_segment_points()
                if start_point is None:
                    self._set_state(ADVANCE_STROKE)
                else:
                    self._publish_pen(pen_up_pos)
                    self._publish_drawing_active(False)
                    cmd = self._tracking_cmd(
                        start_point[0],
                        start_point[1],
                        target_theta,
                        reposition_speed,
                    )
                    self._cmd_pub.publish(cmd)
                    if self._segment_complete(start_point[0], start_point[1], target_theta):
                        self._probe_retries = 0
                        self._lost_contact_cycles = 0
                        self._draw_pen_target = None
                        if stroke['draw']:
                            self._enter_probe_state(pen_up_pos)
                        else:
                            self._set_state(DRAW_SEGMENT)

        elif self._state == PEN_PROBE:
            self._publish_drawing_active(False)  
            self._probe_step()

        elif self._state == PEN_SETTLE:
            if not self._pen_data_fresh(pen_contact_timeout_sec):
                self._fail_execution(
                    'Pen contact/gap data stale during PEN_SETTLE; stopping safely.'
                )
                self._enabled_last = True
                return
            self._publish_zero_twist()
            self._publish_drawing_active(False) 
            if self._draw_pen_target is None:
                self._draw_pen_target = float(self.get_parameter('pen_down_min_pos').value)
            self._publish_pen(self._draw_pen_target)
            self._pen_settle_counter += 1
            if self._pen_settle_counter >= pen_settle_cycles:
                self._set_state(DRAW_SEGMENT)

        elif self._state == DRAW_SEGMENT:
            stroke = self._current_stroke()
            start_point, end_point = self._current_segment_points()
            if stroke is None or start_point is None or end_point is None:
                self._set_state(ADVANCE_STROKE)
            else:
                if stroke['draw']:
                    if not self._pen_data_fresh(pen_contact_timeout_sec):
                        self._fail_execution(
                            'Pen contact/gap data stale during DRAW_SEGMENT; stopping safely.'
                        )
                        self._enabled_last = True
                        return
                    if contact_required:
                        if self._gap_contact_good(pen_contact_timeout_sec):
                            self._lost_contact_cycles = 0
                        elif (
                            self._pen_gap_fresh(pen_contact_timeout_sec)
                            and self._pen_gap > lost_contact_gap_threshold
                        ):
                            self._lost_contact_cycles += 1
                            if (
                                self._lost_contact_cycles
                                >= lost_contact_cycles_before_reprobe
                            ):
                                self.get_logger().warn(
                                    'Contact lost during DRAW_SEGMENT, re-probing.'
                                )
                                self._enter_probe_state(pen_up_pos)
                                return
                        else:
                            self._lost_contact_cycles = 0

                    if self._draw_pen_target is None:
                        self._draw_pen_target = float(
                            self.get_parameter('pen_down_min_pos').value
                        )
                    self._draw_pen_target = self._update_draw_pen_target(
                        self._draw_pen_target
                    )
                    self._publish_pen(self._draw_pen_target)

                    draw_start_delay = max(
                        0, int(self.get_parameter('draw_start_delay_cycles').value)
                    )
                    if self._draw_segment_cycles >= draw_start_delay:
                        self._publish_drawing_active(True)
                    else:
                        self._publish_drawing_active(False)
                    self._draw_segment_cycles += 1

                    speed_cap = draw_speed
                else:
                    self._publish_pen(pen_up_pos)
                    self._publish_drawing_active(False) 
                    speed_cap = reposition_speed

                cmd = self._tracking_cmd(
                    end_point[0],
                    end_point[1],
                    target_theta,
                    speed_cap,
                )
                self._cmd_pub.publish(cmd)

                if self._segment_complete(end_point[0], end_point[1], target_theta):
                    if self._segment_is_last():
                        if stroke['draw']:
                            self._start_pen_up_wait(ADVANCE_STROKE)
                        else:
                            self._set_state(ADVANCE_STROKE)
                    else:
                        self._set_state(CORNER_SETTLE)

        elif self._state == CORNER_SETTLE:
            self._publish_zero_twist()
            self._publish_drawing_active(False)
            if self._draw_pen_target is not None:
                self._publish_pen(self._draw_pen_target)
            self._corner_settle_counter += 1
            corner_settle_cycles = max(
                1, int(self.get_parameter('corner_settle_cycles').value)
            )
            if self._corner_settle_counter >= corner_settle_cycles:
                self._set_state(ADVANCE_SEGMENT)

        elif self._state == PEN_UP:
            if not self._pen_data_fresh(pen_contact_timeout_sec):
                self._fail_execution(
                    'Pen contact/gap data stale during PEN_UP verification; stopping safely.'
                )
                self._enabled_last = True
                return
            self._handle_pen_up_wait()

        elif self._state == ADVANCE_SEGMENT:
            stroke = self._current_stroke()
            if stroke is None:
                self._set_state(ADVANCE_STROKE)
            else:
                self._segment_index += 1
                if self._segment_index >= len(stroke['points']) - 1:
                    self._set_state(ADVANCE_STROKE)
                else:
                    self._lost_contact_cycles = 0
                    self._set_state(DRAW_SEGMENT)

        elif self._state == ADVANCE_STROKE:
            self._stroke_index += 1
            self._segment_index = 0
            self._probe_retries = 0
            self._lost_contact_cycles = 0
            self._draw_pen_target = None
            if self._stroke_index >= len(self._current_plan['strokes']):
                self._set_status('done')
                self._set_state(DONE)
            else:
                self._set_state(MOVE_TO_STROKE_START)

        elif self._state == DONE:
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)
            self._publish_drawing_active(True)

        self._enabled_last = True


def main(args=None):
    rclpy.init(args=args)
    node = StrokeExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
