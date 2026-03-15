"""Generic stroke executor for wall-climbing robot drawing tasks.

Executes stroke plans with line and polyline primitives. Reuses proven
contact-aware pen control logic from the existing line_demo_controller.
"""

import json
import math

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D, Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String


# State machine states
IDLE = 'IDLE'
LOAD_PLAN = 'LOAD_PLAN'
MOVE_TO_STROKE_START = 'MOVE_TO_STROKE_START'
PEN_PROBE = 'PEN_PROBE'
PEN_SETTLE = 'PEN_SETTLE'
DRAW_SEGMENT = 'DRAW_SEGMENT'
PEN_UP = 'PEN_UP'
ADVANCE_SEGMENT = 'ADVANCE_SEGMENT'
ADVANCE_STROKE = 'ADVANCE_STROKE'
DONE = 'DONE'
ERROR = 'ERROR'


def _clamp(v, lo, hi):
    """Clamp value v to range [lo, hi]."""
    return max(lo, min(hi, v))


def _wrap_to_pi(a):
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class StrokeExecutor(Node):
    """Execute generic stroke plans on the wall-climbing robot."""

    def __init__(self):
        super().__init__('stroke_executor')

        # Declare all parameters matching line_demo_controller interface
        self.declare_parameter('enabled', False)
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

        self.declare_parameter('pen_probe_step', 0.006)
        self.declare_parameter('pen_probe_period_cycles', 1)
        self.declare_parameter('pen_settle_cycles', 4)
        self.declare_parameter('pen_contact_timeout_sec', 1.5)
        self.declare_parameter('pen_pose_timeout_sec', 0.5)
        self.declare_parameter('contact_gap_min', -0.0018)
        self.declare_parameter('contact_gap_max', 0.0018)
        self.declare_parameter('lost_contact_cycles_before_reprobe', 8)
        self.declare_parameter('lost_contact_gap_threshold', 0.004)
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

        # Pen board pose (for x/y alignment)
        self._pen_x = None
        self._pen_y = None
        self._pen_pose_stamp = None

        # Board geometry
        self._board = None

        # Contact/gap state
        self._pen_contact = False
        self._pen_gap = float('nan')
        self._pen_contact_stamp = None
        self._pen_gap_stamp = None

        # FSM state
        self._state = IDLE
        self._enabled_last = False

        # Stroke plan execution state
        self._plan = None
        self._current_stroke_idx = 0
        self._current_segment_idx = 0
        self._current_target_x = None
        self._current_target_y = None

        # Pen probe/settle state
        self._probe_target = None
        self._probe_cycle_counter = 0
        self._pen_settle_counter = 0
        self._draw_pen_target = None
        self._lost_contact_cycles = 0
        self._pen_lift_state_start_sec = None

        # Subscriptions
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
        self.create_subscription(
            String, '/wall_climber/stroke_plan', self._stroke_plan_cb, 10
        )

        # Publishers
        self._cmd_pub = self.create_publisher(Twist, '/wall_climber/cmd_vel_auto', 10)
        self._pen_pub = self.create_publisher(Float64, '/wall_climber/pen_target', 10)
        self._status_pub = self.create_publisher(String, '/wall_climber/stroke_executor_status', 10)

        # Timer
        self._timer = self.create_timer(0.05, self._on_timer)  # 20 Hz
        self.get_logger().info('Stroke executor ready (enabled=false).')

    def _pose_cb(self, msg):
        """Robot body pose callback."""
        self._pose = msg
        self._pose_stamp = self.get_clock().now()

    def _pen_pose_cb(self, msg):
        """Pen pose callback."""
        self._pen_x = float(msg.point.x)
        self._pen_y = float(msg.point.y)
        self._pen_pose_stamp = self.get_clock().now()

    def _board_info_cb(self, msg):
        """Board info callback."""
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
        """Pen contact callback."""
        self._pen_contact = bool(msg.data)
        self._pen_contact_stamp = self.get_clock().now()

    def _pen_gap_cb(self, msg):
        """Pen gap callback."""
        self._pen_gap = float(msg.data)
        self._pen_gap_stamp = self.get_clock().now()

    def _stroke_plan_cb(self, msg):
        """Stroke plan callback - validate and store plan."""
        try:
            plan = json.loads(msg.data)
            if not self._validate_plan(plan):
                self.get_logger().error('Invalid stroke plan received - rejected')
                self._publish_status('error')
                return

            self._plan = plan
            self.get_logger().info(f'Loaded stroke plan with {len(plan["strokes"])} strokes')

            # Reset execution state
            self._current_stroke_idx = 0
            self._current_segment_idx = 0

            # Auto-start if enabled
            enabled = bool(self.get_parameter('enabled').value)
            if enabled and self._state in (IDLE, DONE, ERROR):
                self._set_state(LOAD_PLAN)

        except json.JSONDecodeError as exc:
            self.get_logger().error(f'Failed to parse stroke plan JSON: {exc}')
            self._publish_status('error')
        except Exception as exc:
            self.get_logger().error(f'Error processing stroke plan: {exc}')
            self._publish_status('error')

    def _validate_plan(self, plan):
        """Validate stroke plan structure and bounds."""
        if not isinstance(plan, dict):
            self.get_logger().error('Plan must be a JSON object')
            return False

        # Check frame
        if plan.get('frame') != 'board':
            self.get_logger().error('Plan frame must be "board"')
            return False

        # Check strokes
        if 'strokes' not in plan or not isinstance(plan['strokes'], list):
            self.get_logger().error('Plan must contain "strokes" list')
            return False

        if len(plan['strokes']) == 0:
            self.get_logger().warn('Plan contains no strokes')
            return True  # Empty plan is valid, just does nothing

        # Validate each stroke
        for i, stroke in enumerate(plan['strokes']):
            if not isinstance(stroke, dict):
                self.get_logger().error(f'Stroke {i} must be an object')
                return False

            # Check type
            stroke_type = stroke.get('type')
            if stroke_type not in ('line', 'polyline'):
                self.get_logger().error(f'Stroke {i} type must be "line" or "polyline"')
                return False

            # Check points
            points = stroke.get('points')
            if not isinstance(points, list):
                self.get_logger().error(f'Stroke {i} must have "points" list')
                return False

            if stroke_type == 'line' and len(points) != 2:
                self.get_logger().error(f'Stroke {i} type "line" must have exactly 2 points')
                return False

            if stroke_type == 'polyline' and len(points) < 2:
                self.get_logger().error(f'Stroke {i} type "polyline" must have at least 2 points')
                return False

            # Check each point
            for j, pt in enumerate(points):
                if not isinstance(pt, list) or len(pt) != 2:
                    self.get_logger().error(f'Stroke {i} point {j} must be [x, y]')
                    return False

                try:
                    x, y = float(pt[0]), float(pt[1])
                except (ValueError, TypeError):
                    self.get_logger().error(f'Stroke {i} point {j} must contain numbers')
                    return False

                # Validate bounds if board info available
                if self._board is not None:
                    if not self._point_in_bounds(x, y):
                        self.get_logger().error(
                            f'Stroke {i} point {j} ({x:.3f}, {y:.3f}) is outside writable bounds'
                        )
                        return False

            # Check draw flag
            if 'draw' not in stroke:
                self.get_logger().error(f'Stroke {i} must have "draw" boolean')
                return False

        return True

    def _point_in_bounds(self, x, y):
        """Check if point is within writable board bounds."""
        x_min = float(self._board['writable_x_min'])
        x_max = float(self._board['writable_x_max'])
        y_min = float(self._board['writable_y_min'])
        y_max = float(self._board['writable_y_max'])
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _publish_pen(self, value):
        """Publish pen target position."""
        msg = Float64()
        msg.data = float(value)
        self._pen_pub.publish(msg)

    def _publish_zero_twist(self):
        """Publish zero velocity command."""
        self._cmd_pub.publish(Twist())

    def _publish_status(self, status):
        """Publish executor status."""
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _set_state(self, new_state):
        """Transition to new state."""
        if self._state == new_state:
            return
        self._state = new_state
        self.get_logger().info(f'State: {new_state}')

        # Reset state-specific counters
        if new_state == PEN_PROBE:
            self._probe_cycle_counter = 0
        if new_state == PEN_SETTLE:
            self._pen_settle_counter = 0
        if new_state == DRAW_SEGMENT:
            self._lost_contact_cycles = 0

    def _now_sec(self):
        """Get current time in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose_fresh(self, timeout_sec):
        """Check if robot pose is fresh."""
        if self._pose is None or self._pose_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pose_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_pose_fresh(self, timeout_sec):
        """Check if pen pose is fresh."""
        if self._pen_x is None or self._pen_y is None or self._pen_pose_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pen_pose_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_contact_fresh(self, timeout_sec):
        """Check if pen contact data is fresh."""
        if self._pen_contact_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pen_contact_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_gap_fresh(self, timeout_sec):
        """Check if pen gap data is fresh."""
        if self._pen_gap_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pen_gap_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_data_fresh(self, timeout_sec):
        """Check if both contact and gap data are fresh."""
        return self._pen_contact_fresh(timeout_sec) and self._pen_gap_fresh(timeout_sec)

    def _gap_contact_good(self, timeout_sec):
        """Check if pen gap indicates good contact."""
        if not self._pen_gap_fresh(timeout_sec):
            return False
        g = self._pen_gap
        g_min = float(self.get_parameter('contact_gap_min').value)
        g_max = float(self.get_parameter('contact_gap_max').value)
        return g_min <= g <= g_max

    def _reset_execution(self):
        """Reset execution state."""
        self._state = IDLE
        self._current_stroke_idx = 0
        self._current_segment_idx = 0
        self._current_target_x = None
        self._current_target_y = None
        self._probe_target = None
        self._probe_cycle_counter = 0
        self._pen_settle_counter = 0
        self._draw_pen_target = None
        self._lost_contact_cycles = 0
        self._pen_lift_state_start_sec = None

    def _tracking_cmd(self, target_x, target_y, target_theta, x_cap):
        """Compute tracking command to move pen to target position.

        Uses pen pose for x/y alignment and robot pose for heading.
        Reuses proven logic from line_demo_controller.
        """
        k_y = float(self.get_parameter('k_y').value)
        k_theta = float(self.get_parameter('k_theta').value)
        omega_sign = float(self.get_parameter('omega_sign').value)
        max_lat = float(self.get_parameter('max_lateral_cmd').value)
        max_ang = float(self.get_parameter('max_angular_cmd').value)

        # Pen pose drives positional control
        x = float(self._pen_x)
        y = float(self._pen_y)
        # Robot body pose drives heading correction
        theta = float(self._pose.theta)

        downward_error = y - target_y
        theta_error = _wrap_to_pi(target_theta - theta)

        cmd = Twist()
        cmd.linear.y = _clamp(k_y * downward_error, -max_lat, max_lat)
        cmd.angular.z = _clamp(omega_sign * k_theta * theta_error, -max_ang, max_ang)

        # Safety gate: reduce x motion when y/theta errors are large
        if abs(downward_error) > 0.03 or abs(theta_error) > 0.08:
            cmd.linear.x = 0.0
            return cmd

        x_error = target_x - x
        cmd.linear.x = _clamp(2.0 * x_error, -float(x_cap), float(x_cap))

        return cmd

    def _handle_pen_probe(self):
        """Handle pen probing to find contact.

        Reuses proven probe logic from line_demo_controller.
        """
        pen_down_min = float(self.get_parameter('pen_down_min_pos').value)
        pen_down_max = float(self.get_parameter('pen_down_max_pos').value)
        pen_up_pos = float(self.get_parameter('pen_up_pos').value)
        probe_step = float(self.get_parameter('pen_probe_step').value)
        probe_period = max(1, int(self.get_parameter('pen_probe_period_cycles').value))
        contact_timeout_sec = float(self.get_parameter('pen_contact_timeout_sec').value)
        contact_gap_min = float(self.get_parameter('contact_gap_min').value)
        draw_pen_extra_depth = float(self.get_parameter('draw_pen_extra_depth').value)

        if self._probe_target is None:
            self._probe_target = pen_up_pos

        # Check if we found good contact
        if self._gap_contact_good(contact_timeout_sec):
            self._draw_pen_target = max(pen_down_max, self._probe_target - draw_pen_extra_depth)
            self._set_state(PEN_SETTLE)
            return

        self._publish_zero_twist()

        # Gap-aware probing
        if self._pen_gap_fresh(contact_timeout_sec) and self._pen_gap < contact_gap_min:
            # Too deep - lift slightly
            self._probe_target = min(pen_up_pos, self._probe_target + probe_step)
            self._probe_cycle_counter = 0
        else:
            # Probe downward
            if self._probe_target > pen_down_min:
                self._probe_target = pen_down_min
            else:
                self._probe_cycle_counter += 1
                if self._probe_cycle_counter >= probe_period:
                    self._probe_target = max(pen_down_max, self._probe_target - probe_step)
                    self._probe_cycle_counter = 0

        self._publish_pen(self._probe_target)

        # Check for probe failure
        if self._probe_target <= pen_down_max and not self._gap_contact_good(contact_timeout_sec):
            self.get_logger().error(
                f'Pen probe failed (gap={self._pen_gap:.4f} m). Stopping safely.'
            )
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)
            self._set_state(ERROR)
            self._publish_status('error')

    def _handle_pen_settle(self):
        """Handle pen settle period after probe."""
        pen_settle_cycles = max(1, int(self.get_parameter('pen_settle_cycles').value))

        self._publish_zero_twist()
        if self._draw_pen_target is None:
            self._draw_pen_target = float(self.get_parameter('pen_down_min_pos').value)
        self._publish_pen(self._draw_pen_target)

        self._pen_settle_counter += 1
        if self._pen_settle_counter >= pen_settle_cycles:
            self._set_state(DRAW_SEGMENT)

    def _handle_pen_up_wait(self):
        """Handle waiting for pen to lift."""
        pen_up_pos = float(self.get_parameter('pen_up_pos').value)
        pen_clear_gap = float(self.get_parameter('pen_clear_gap').value)
        pen_lift_timeout_sec = float(self.get_parameter('pen_lift_timeout_sec').value)
        pen_contact_timeout_sec = float(self.get_parameter('pen_contact_timeout_sec').value)

        self._publish_zero_twist()
        self._publish_pen(pen_up_pos)

        # Check if pen is lifted
        lifted = (
            self._pen_data_fresh(pen_contact_timeout_sec)
            and (not self._pen_contact)
            and (self._pen_gap > pen_clear_gap)
        )
        if lifted:
            self._set_state(ADVANCE_SEGMENT)
            return

        # Initialize lift timer
        if self._pen_lift_state_start_sec is None:
            self._pen_lift_state_start_sec = self._now_sec()
            return

        # Check for timeout
        if (self._now_sec() - self._pen_lift_state_start_sec) > pen_lift_timeout_sec:
            self.get_logger().warn('Pen lift timeout; continuing safely.')
            self._set_state(ADVANCE_SEGMENT)

    def _update_draw_pen_target(self):
        """Keep draw target near calibrated contact gap during drawing."""
        pen_contact_timeout_sec = float(self.get_parameter('pen_contact_timeout_sec').value)
        contact_gap_min = float(self.get_parameter('contact_gap_min').value)
        contact_gap_max = float(self.get_parameter('contact_gap_max').value)
        recover_step = float(self.get_parameter('draw_pen_recover_step').value)
        pen_down_max = float(self.get_parameter('pen_down_max_pos').value)
        pen_up_pos = float(self.get_parameter('pen_up_pos').value)

        if not self._pen_gap_fresh(pen_contact_timeout_sec):
            return

        if self._pen_gap > contact_gap_max:
            self._draw_pen_target -= recover_step
        elif self._pen_gap < contact_gap_min:
            self._draw_pen_target += recover_step

        self._draw_pen_target = _clamp(self._draw_pen_target, pen_down_max, pen_up_pos)

    def _get_current_stroke(self):
        """Get current stroke from plan."""
        if self._plan is None or self._current_stroke_idx >= len(self._plan['strokes']):
            return None
        return self._plan['strokes'][self._current_stroke_idx]

    def _get_segment_target(self):
        """Get target point for current segment."""
        stroke = self._get_current_stroke()
        if stroke is None:
            return None, None

        points = stroke['points']
        if self._current_segment_idx >= len(points):
            return None, None

        pt = points[self._current_segment_idx]
        return float(pt[0]), float(pt[1])

    def _on_timer(self):
        """Main timer callback - FSM executor."""
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
        lost_contact_gap_threshold = float(self.get_parameter('lost_contact_gap_threshold').value)
        lost_contact_cycles_before_reprobe = max(
            1, int(self.get_parameter('lost_contact_cycles_before_reprobe').value)
        )
        publish_zero_on_stop = bool(self.get_parameter('publish_zero_on_stop').value)

        # Handle disable
        if not enabled:
            if self._enabled_last and publish_zero_on_stop:
                self._publish_zero_twist()
                self._publish_pen(pen_up_pos)
                self._publish_status('idle')
            self._reset_execution()
            self._enabled_last = False
            return

        # Handle enable transition
        if enabled and not self._enabled_last:
            self._reset_execution()
            self.get_logger().info('Enabled - waiting for plan and pose data')
            self._publish_status('idle')

        # Check for required data
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

        pen_x = float(self._pen_x)
        pen_y = float(self._pen_y)
        theta = float(self._pose.theta)
        theta_error = _wrap_to_pi(target_theta - theta)

        # FSM states
        if self._state == IDLE:
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)
            self._publish_status('idle')

        elif self._state == LOAD_PLAN:
            if self._plan is None or len(self._plan['strokes']) == 0:
                self.get_logger().warn('No valid plan to execute')
                self._set_state(IDLE)
                return

            self._current_stroke_idx = 0
            self._current_segment_idx = 0
            self._set_state(MOVE_TO_STROKE_START)
            self._publish_status('running')

        elif self._state == MOVE_TO_STROKE_START:
            stroke = self._get_current_stroke()
            if stroke is None:
                self._set_state(DONE)
                return

            # Get first point of stroke
            pt = stroke['points'][0]
            target_x, target_y = float(pt[0]), float(pt[1])

            self._publish_pen(pen_up_pos)
            cmd = self._tracking_cmd(target_x, target_y, target_theta, reposition_speed)
            self._cmd_pub.publish(cmd)

            # Check arrival
            if (abs(pen_x - target_x) < pos_tol_x and
                abs(pen_y - target_y) < pos_tol_y and
                abs(theta_error) < theta_tol):

                # Check if we need to draw
                if stroke['draw']:
                    self._current_segment_idx = 1  # Next target is second point
                    self._probe_target = pen_up_pos
                    self._set_state(PEN_PROBE)
                else:
                    # Move without drawing - advance to next segment
                    self._current_segment_idx = 1
                    if self._current_segment_idx >= len(stroke['points']):
                        self._set_state(ADVANCE_STROKE)
                    else:
                        # Continue moving to next point
                        pass  # Stay in MOVE_TO_STROKE_START

        elif self._state == PEN_PROBE:
            self._handle_pen_probe()

        elif self._state == PEN_SETTLE:
            self._handle_pen_settle()

        elif self._state == DRAW_SEGMENT:
            stroke = self._get_current_stroke()
            if stroke is None:
                self._set_state(DONE)
                return

            target_x, target_y = self._get_segment_target()
            if target_x is None:
                # Finished all segments in stroke
                self._pen_lift_state_start_sec = None
                self._set_state(PEN_UP)
                return

            # Monitor contact during drawing
            if self._gap_contact_good(pen_contact_timeout_sec):
                self._lost_contact_cycles = 0
            elif (self._pen_gap_fresh(pen_contact_timeout_sec) and
                  self._pen_gap > lost_contact_gap_threshold):
                self._lost_contact_cycles += 1
                if self._lost_contact_cycles >= lost_contact_cycles_before_reprobe:
                    self.get_logger().warn('Contact lost during drawing, re-probing')
                    self._probe_target = pen_up_pos
                    # Go back to start of current segment
                    if self._current_segment_idx > 0:
                        self._current_segment_idx -= 1
                    self._set_state(PEN_PROBE)
                    return
            else:
                self._lost_contact_cycles = 0

            # Update pen depth
            if self._draw_pen_target is None:
                self._draw_pen_target = float(self.get_parameter('pen_down_min_pos').value)
            self._update_draw_pen_target()
            self._publish_pen(self._draw_pen_target)

            # Track to target
            cmd = self._tracking_cmd(target_x, target_y, target_theta, draw_speed)
            self._cmd_pub.publish(cmd)

            # Check arrival
            if abs(pen_x - target_x) < pos_tol_x and abs(pen_y - target_y) < pos_tol_y:
                # Move to next segment
                self._current_segment_idx += 1
                if self._current_segment_idx >= len(stroke['points']):
                    # Finished stroke - lift pen
                    self._pen_lift_state_start_sec = None
                    self._set_state(PEN_UP)
                # else: continue drawing to next point

        elif self._state == PEN_UP:
            self._handle_pen_up_wait()

        elif self._state == ADVANCE_SEGMENT:
            # Reset pen lift timer
            self._pen_lift_state_start_sec = None

            # Move to next stroke
            self._set_state(ADVANCE_STROKE)

        elif self._state == ADVANCE_STROKE:
            self._current_stroke_idx += 1
            self._current_segment_idx = 0

            if self._current_stroke_idx >= len(self._plan['strokes']):
                self._set_state(DONE)
            else:
                self._set_state(MOVE_TO_STROKE_START)

        elif self._state == DONE:
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)
            self._publish_status('done')

        elif self._state == ERROR:
            self._publish_zero_twist()
            self._publish_pen(pen_up_pos)
            self._publish_status('error')

        self._enabled_last = True


def main(args=None):
    """Main entry point."""
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
