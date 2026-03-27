"""Generic board-aware stroke executor for Artie.

This node executes JSON stroke plans in board coordinates using the same
contact-aware drawing behavior proven in line_demo_controller.
"""

from dataclasses import dataclass
import json
import math
from typing import Any

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D, Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String


# Timer frequency in Hz. All cycle-based parameters are expressed in ticks at
# this rate. If you change TIMER_HZ, update the declare_parameter defaults below
# so that the real-time durations stay the same.
TIMER_HZ = 60.0

IDLE = 'IDLE'
MOVE_TO_STROKE_START = 'MOVE_TO_STROKE_START'
PEN_PROBE = 'PEN_PROBE'
PEN_SETTLE = 'PEN_SETTLE'
DRAW_SEGMENT = 'DRAW_SEGMENT'
CORNER_SETTLE = 'CORNER_SETTLE'
PEN_UP = 'PEN_UP'
ADVANCE_SEGMENT = 'ADVANCE_SEGMENT'
ADVANCE_STROKE = 'ADVANCE_STROKE'
DONE = 'DONE'


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class PathPrimitive:
    """Internal execution primitive derived from one normalized external stroke."""

    kind: str
    draw: bool
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ExecutionPath:
    """Internal path model consumed by the FSM after validation succeeds."""

    frame: str
    primitives: tuple[PathPrimitive, ...]


@dataclass(frozen=True)
class TickParams:
    enabled: bool
    draw_speed: float
    reposition_speed: float
    target_theta: float
    k_y: float
    k_theta: float
    omega_sign: float
    max_lateral_cmd: float
    max_angular_cmd: float
    pos_tol_x: float
    pos_tol_y: float
    theta_tol: float
    contact_required_for_drawing: bool
    pen_probe_step: float
    pen_probe_period_cycles: int
    pen_settle_cycles: int
    corner_settle_cycles: int
    draw_start_delay_cycles: int
    pen_contact_timeout_sec: float
    pen_pose_timeout_sec: float
    pose_timeout_sec: float
    lost_contact_cycles_before_reprobe: int
    max_probe_retries_per_line: int
    pen_up_pos: float
    pen_clear_gap: float
    pen_lift_timeout_sec: float
    pen_down_min_pos: float
    pen_down_max_pos: float
    publish_zero_on_stop: bool
    publish_debug_telemetry: bool


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
        self.declare_parameter('pen_probe_step', 0.0005)
        # 1 tick at 60 Hz ≈ 16.7 ms per probe step.
        self.declare_parameter('pen_probe_period_cycles', 1)
        # 12 ticks at 60 Hz ≈ 200 ms  (was 4 ticks at 20 Hz)
        self.declare_parameter('pen_settle_cycles', 12)
        # 9 ticks at 60 Hz ≈ 150 ms  (was 3 ticks at 20 Hz)
        self.declare_parameter('corner_settle_cycles', 9)
        self.declare_parameter('draw_start_delay_cycles', 0)
        self.declare_parameter('pen_contact_timeout_sec', 1.5)
        self.declare_parameter('pen_pose_timeout_sec', 0.5)
        # Legacy compatibility parameters from the old gap-window contact model.
        # Collision-first contact uses supervisor-derived pen_contact directly.
        self.declare_parameter('contact_gap_min', -0.0018)
        self.declare_parameter('contact_gap_max', 0.0018)
        # 24 ticks at 60 Hz ≈ 400 ms  (was 8 ticks at 20 Hz)
        self.declare_parameter('lost_contact_cycles_before_reprobe', 24)
        # Legacy compatibility parameter kept only to avoid breaking runtime configs.
        self.declare_parameter('lost_contact_gap_threshold', 0.004)
        self.declare_parameter('max_probe_retries_per_line', 3)
        # Patch-3 keeps stroke starts shallow: no extra dig beyond first contact.
        self.declare_parameter('draw_pen_extra_depth', 0.0)
        # Patch-3 also disables ongoing depth recovery to avoid re-pushing inward.
        self.declare_parameter('draw_pen_recover_step', 0.0)

        self.declare_parameter('pen_up_pos', 0.018)
        self.declare_parameter('pen_clear_gap', 0.004)
        self.declare_parameter('pen_lift_timeout_sec', 1.5)
        # More-negative pen targets mean deeper physical insertion toward the board.
        self.declare_parameter('pen_down_min_pos', -0.010)
        self.declare_parameter('pen_down_max_pos', -0.030)

        self.declare_parameter('publish_zero_on_stop', True)
        self.declare_parameter('pose_timeout_sec', 0.5)
        self.declare_parameter('publish_debug_telemetry', False)

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
        self._drawing_active = False

        self._pending_external_plan = None
        self._current_external_plan = None
        self._current_exec_path = None
        self._primitive_index = 0
        self._segment_index = 0

        self._probe_target = None
        self._probe_cycle_counter = 0
        self._pen_settle_counter = 0
        self._corner_settle_counter = 0
        self._corner_stable_counter = 0
        self._draw_pen_target = None
        self._lost_contact_cycles = 0
        self._probe_retries = 0
        self._pen_lift_state_start_sec = None
        self._next_state_after_pen_up = None
        self._draw_segment_cycles = 0
        self._post_corner_draw_delay_cycles = 0
        self._debug_tick_counter = 0
        self._debug_transition_reason = None
        self._tick_params = self._read_tick_params()

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
        self._debug_pub = self.create_publisher(
            String, '/wall_climber/stroke_executor_debug', 10
        )

        self._timer = self.create_timer(1.0 / TIMER_HZ, self._on_timer)
        self._set_status('idle')
        self.get_logger().info('Stroke executor ready (enabled=false).')

    def _read_tick_params(self) -> TickParams:
        return TickParams(
            enabled=bool(self.get_parameter('enabled').value),
            draw_speed=float(self.get_parameter('draw_speed').value),
            reposition_speed=float(self.get_parameter('reposition_speed').value),
            target_theta=float(self.get_parameter('target_theta').value),
            k_y=float(self.get_parameter('k_y').value),
            k_theta=float(self.get_parameter('k_theta').value),
            omega_sign=float(self.get_parameter('omega_sign').value),
            max_lateral_cmd=float(self.get_parameter('max_lateral_cmd').value),
            max_angular_cmd=float(self.get_parameter('max_angular_cmd').value),
            pos_tol_x=float(self.get_parameter('pos_tol_x').value),
            pos_tol_y=float(self.get_parameter('pos_tol_y').value),
            theta_tol=float(self.get_parameter('theta_tol').value),
            contact_required_for_drawing=bool(
                self.get_parameter('contact_required_for_drawing').value
            ),
            pen_probe_step=float(self.get_parameter('pen_probe_step').value),
            pen_probe_period_cycles=max(
                1, int(self.get_parameter('pen_probe_period_cycles').value)
            ),
            pen_settle_cycles=max(1, int(self.get_parameter('pen_settle_cycles').value)),
            corner_settle_cycles=max(
                1, int(self.get_parameter('corner_settle_cycles').value)
            ),
            draw_start_delay_cycles=max(
                0, int(self.get_parameter('draw_start_delay_cycles').value)
            ),
            pen_contact_timeout_sec=float(
                self.get_parameter('pen_contact_timeout_sec').value
            ),
            pen_pose_timeout_sec=float(self.get_parameter('pen_pose_timeout_sec').value),
            pose_timeout_sec=float(self.get_parameter('pose_timeout_sec').value),
            lost_contact_cycles_before_reprobe=max(
                1, int(self.get_parameter('lost_contact_cycles_before_reprobe').value)
            ),
            max_probe_retries_per_line=max(
                0, int(self.get_parameter('max_probe_retries_per_line').value)
            ),
            pen_up_pos=float(self.get_parameter('pen_up_pos').value),
            pen_clear_gap=float(self.get_parameter('pen_clear_gap').value),
            pen_lift_timeout_sec=float(self.get_parameter('pen_lift_timeout_sec').value),
            pen_down_min_pos=float(self.get_parameter('pen_down_min_pos').value),
            pen_down_max_pos=float(self.get_parameter('pen_down_max_pos').value),
            publish_zero_on_stop=bool(self.get_parameter('publish_zero_on_stop').value),
            publish_debug_telemetry=bool(
                self.get_parameter('publish_debug_telemetry').value
            ),
        )

    def _pose_cb(self, msg: Pose2D) -> None:
        self._pose = msg
        self._pose_stamp = self.get_clock().now()

    def _pen_pose_cb(self, msg: PointStamped) -> None:
        self._pen_x = float(msg.point.x)
        self._pen_y = float(msg.point.y)
        self._pen_pose_stamp = self.get_clock().now()

    def _pen_contact_cb(self, msg: Bool) -> None:
        self._pen_contact = bool(msg.data)
        self._pen_contact_stamp = self.get_clock().now()

    def _pen_gap_cb(self, msg: Float64) -> None:
        self._pen_gap = float(msg.data)
        self._pen_gap_stamp = self.get_clock().now()

    def _board_info_cb(self, msg: String) -> None:
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

    def _plan_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self._reject_plan(f'Malformed stroke plan JSON: {exc}')
            return

        normalized, error = self._normalize_plan(payload)
        if error is not None:
            self._reject_plan(error)
            return

        self._pending_external_plan = normalized
        if self._board is None:
            self.get_logger().warn(
                'Received structurally valid stroke plan, but board_info is not ready yet; '
                'deferring writable-area validation.'
            )
            return

        self._finalize_pending_plan()

    def _normalize_plan(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
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

    def _segment_count_total(self, plan: dict[str, Any]) -> int:
        return sum(max(len(stroke['points']) - 1, 0) for stroke in plan['strokes'])

    def _execution_segment_count(self, exec_path: ExecutionPath) -> int:
        return sum(max(len(primitive.points) - 1, 0) for primitive in exec_path.primitives)

    def _build_execution_path(self, external_plan: dict[str, Any]) -> ExecutionPath:
        primitives = []
        for stroke in external_plan['strokes']:
            if stroke['type'] == 'line':
                kind = 'line' if stroke['draw'] else 'travel'
            else:
                kind = 'polyline' if stroke['draw'] else 'travel'

            primitives.append(
                PathPrimitive(
                    kind=kind,
                    draw=bool(stroke['draw']),
                    points=tuple(stroke['points']),
                )
            )

        return ExecutionPath(
            frame=str(external_plan['frame']),
            primitives=tuple(primitives),
        )

    def _segment_is_axis_aligned(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        epsilon: float = 1e-6,
    ) -> bool:
        dx = abs(float(end_point[0]) - float(start_point[0]))
        dy = abs(float(end_point[1]) - float(start_point[1]))
        return dx <= epsilon or dy <= epsilon

    def _point_inside_writable(self, point: tuple[float, float]) -> bool:
        if self._board is None:
            return False
        x, y = point
        return (
            float(self._board['writable_x_min']) <= x <= float(self._board['writable_x_max'])
            and float(self._board['writable_y_min']) <= y <= float(self._board['writable_y_max'])
        )

    def _validate_plan_points(self, plan: dict[str, Any]) -> str | None:
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
        if self._pending_external_plan is None:
            return

        error = self._validate_plan_points(self._pending_external_plan)
        if error is not None:
            self._pending_external_plan = None
            self._reject_plan(error)
            return

        self._current_external_plan = self._pending_external_plan
        self._current_exec_path = self._build_execution_path(
            self._current_external_plan
        )
        self._pending_external_plan = None
        self._reset_execution()

        stroke_count = len(self._current_exec_path.primitives)
        segment_count = self._execution_segment_count(self._current_exec_path)
        first_stroke_type = (
            self._current_external_plan['strokes'][0]['type']
            if stroke_count > 0
            else 'none'
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

    def _reject_plan(self, reason: str) -> None:
        self.get_logger().warn(reason)
        self._publish_zero_twist()
        self._publish_pen(float(self.get_parameter('pen_up_pos').value))
        self._publish_drawing_active(False)
        self._set_status('error')
        self._set_state(DONE)

    def _publish_pen(self, value: float) -> None:
        msg = Float64()
        msg.data = float(value)
        self._pen_pub.publish(msg)

    def _publish_drawing_active(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        self._drawing_active = bool(active)
        self._drawing_active_pub.publish(msg)

    def _publish_zero_twist(self) -> None:
        self._cmd_pub.publish(Twist())

    def _set_status(self, text: str) -> None:
        if self._status == text:
            return
        self._status = text
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)

    def _set_state(self, new_state: str) -> None:
        if self._state == new_state:
            return
        previous_state = self._state
        self._state = new_state
        self._debug_transition_reason = f'state:{new_state}'
        self.get_logger().info(
            f'entered {new_state} '
            f'(primitive={self._primitive_index}, segment={self._segment_index})'
        )
        if new_state == PEN_PROBE:
            self._probe_cycle_counter = 0
        if new_state == PEN_SETTLE:
            self._pen_settle_counter = 0
        if new_state == CORNER_SETTLE:
            self._corner_settle_counter = 0
            self._corner_stable_counter = 0
        if new_state == DRAW_SEGMENT:
            self._lost_contact_cycles = 0
            if previous_state == PEN_SETTLE:
                self._draw_segment_cycles = 0
                self._post_corner_draw_delay_cycles = 0
            elif previous_state == ADVANCE_SEGMENT and self._post_corner_draw_delay_cycles > 0:
                # Re-arm the draw-start delay only for the connected segment that
                # immediately follows a settled polyline corner.
                self._draw_segment_cycles = 0

    def _reset_execution(self) -> None:
        self._state = IDLE
        self._primitive_index = 0
        self._segment_index = 0
        self._probe_target = None
        self._probe_cycle_counter = 0
        self._pen_settle_counter = 0
        self._corner_settle_counter = 0
        self._corner_stable_counter = 0
        self._draw_pen_target = None
        self._lost_contact_cycles = 0
        self._probe_retries = 0
        self._pen_lift_state_start_sec = None
        self._next_state_after_pen_up = None
        self._draw_segment_cycles = 0
        self._post_corner_draw_delay_cycles = 0
        self._drawing_active = False
        self._debug_tick_counter = 0
        self._debug_transition_reason = None

    def _now_sec(self) -> float:
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

    def _pen_data_fresh(self, timeout_sec: float) -> bool:
        return self._pen_contact_fresh(timeout_sec) and self._pen_gap_fresh(timeout_sec)

    def _effective_contact(self, timeout_sec: float) -> bool:
        return self._pen_contact_fresh(timeout_sec) and self._pen_contact

    def _current_primitive(self) -> PathPrimitive | None:
        if self._current_exec_path is None:
            return None
        primitives = self._current_exec_path.primitives
        if not (0 <= self._primitive_index < len(primitives)):
            return None
        return primitives[self._primitive_index]

    def _current_primitive_points(self) -> tuple[tuple[float, float], ...]:
        primitive = self._current_primitive()
        if primitive is None:
            return ()
        return primitive.points

    def _current_primitive_start_point(self) -> tuple[float, float] | None:
        points = self._current_primitive_points()
        if not points:
            return None
        return points[0]

    def _current_primitive_end_point(self) -> tuple[float, float] | None:
        points = self._current_primitive_points()
        if not points:
            return None
        return points[-1]

    def _current_primitive_is_drawn(self) -> bool:
        primitive = self._current_primitive()
        return bool(primitive is not None and primitive.draw)

    def _current_segment_points(
        self,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        points = self._current_primitive_points()
        if not points:
            return None, None
        if not (0 <= self._segment_index < len(points) - 1):
            return None, None
        return points[self._segment_index], points[self._segment_index + 1]

    def _current_segment_is_last(self) -> bool:
        points = self._current_primitive_points()
        if not points:
            return True
        return self._segment_index >= (len(points) - 2)

    def _current_transition_keeps_pen_down(self) -> bool:
        primitive = self._current_primitive()
        return bool(
            primitive is not None
            and primitive.kind == 'polyline'
            and primitive.draw
            and not self._current_segment_is_last()
        )

    def _segment_axis(
        self,
        start_point: tuple[float, float] | None,
        end_point: tuple[float, float] | None,
        epsilon: float = 1e-6,
    ) -> str | None:
        if start_point is None or end_point is None:
            return None
        dx = abs(float(end_point[0]) - float(start_point[0]))
        dy = abs(float(end_point[1]) - float(start_point[1]))
        if dx <= epsilon and dy > epsilon:
            return 'vertical'
        if dy <= epsilon and dx > epsilon:
            return 'horizontal'
        return None

    def _segment_direction_sign(
        self,
        start_point: tuple[float, float] | None,
        end_point: tuple[float, float] | None,
        axis: str | None,
    ) -> float | None:
        if start_point is None or end_point is None or axis is None:
            return None
        if axis == 'horizontal':
            delta = float(end_point[0]) - float(start_point[0])
        else:
            delta = float(end_point[1]) - float(start_point[1])
        if abs(delta) < 1e-9:
            return None
        return 1.0 if delta > 0.0 else -1.0

    def _segment_length(
        self,
        start_point: tuple[float, float] | None,
        end_point: tuple[float, float] | None,
        axis: str | None,
    ) -> float:
        if start_point is None or end_point is None or axis is None:
            return 0.0
        if axis == 'horizontal':
            return abs(float(end_point[0]) - float(start_point[0]))
        return abs(float(end_point[1]) - float(start_point[1]))

    def _along_track_progress(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        tip_x: float,
        tip_y: float,
    ) -> float:
        axis = self._segment_axis(start_point, end_point)
        direction_sign = self._segment_direction_sign(start_point, end_point, axis)
        if axis is None or direction_sign is None:
            return 0.0
        if axis == 'horizontal':
            return direction_sign * (float(tip_x) - float(start_point[0]))
        return direction_sign * (float(tip_y) - float(start_point[1]))

    def _remaining_distance(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        tip_x: float,
        tip_y: float,
    ) -> float:
        axis = self._segment_axis(start_point, end_point)
        direction_sign = self._segment_direction_sign(start_point, end_point, axis)
        if axis is None or direction_sign is None:
            return 0.0
        if axis == 'horizontal':
            return direction_sign * (float(end_point[0]) - float(tip_x))
        return direction_sign * (float(end_point[1]) - float(tip_y))

    def _cross_track_error(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        tip_x: float,
        tip_y: float,
    ) -> float:
        axis = self._segment_axis(start_point, end_point)
        if axis == 'horizontal':
            return float(tip_y) - float(start_point[1])
        if axis == 'vertical':
            return float(tip_x) - float(start_point[0])
        return 0.0

    def _segment_linear_components(
        self,
        axis: str | None,
        along_cmd: float,
        cross_cmd: float,
    ) -> tuple[float, float]:
        if axis == 'horizontal':
            return along_cmd, cross_cmd
        if axis == 'vertical':
            # For vertical segments:
            # - along_cmd > 0 means "move downward on the board"
            # - cross_cmd > 0 means "the tip drifted to the right of the target X"
            # cmd.linear.y uses the opposite sign for board-down motion, while
            # cmd.linear.x must push back toward the segment line.
            return -cross_cmd, -along_cmd
        return 0.0, 0.0

    def _segment_ready(
        self,
        start_point: tuple[float, float] | None,
        end_point: tuple[float, float] | None,
        target_theta: float,
    ) -> bool:
        if (
            start_point is None
            or end_point is None
            or self._pen_x is None
            or self._pen_y is None
            or self._pose is None
        ):
            return False
        axis = self._segment_axis(start_point, end_point)
        if axis is None:
            return False
        pos_tol_x = self._tick_params.pos_tol_x
        pos_tol_y = self._tick_params.pos_tol_y
        theta_tol = self._tick_params.theta_tol
        if axis == 'horizontal':
            along_tol = pos_tol_x
            cross_tol = pos_tol_y
        else:
            along_tol = pos_tol_y
            cross_tol = pos_tol_x
        remaining_distance = self._remaining_distance(
            start_point,
            end_point,
            self._pen_x,
            self._pen_y,
        )
        cross_track_error = self._cross_track_error(
            start_point,
            end_point,
            self._pen_x,
            self._pen_y,
        )
        theta_error = _wrap_to_pi(target_theta - float(self._pose.theta))
        return (
            abs(remaining_distance) < along_tol
            and abs(cross_track_error) < cross_tol
            and abs(theta_error) < theta_tol
        )

    def _corner_hold_ready(
        self,
        corner_x: float,
        corner_y: float,
        target_theta: float,
    ) -> bool:
        if self._pen_x is None or self._pen_y is None or self._pose is None:
            return False
        theta_tol = self._tick_params.theta_tol
        corner_pos_tol_x = min(self._tick_params.pos_tol_x, 0.0015)
        corner_pos_tol_y = min(self._tick_params.pos_tol_y, 0.0015)
        corner_theta_tol = min(theta_tol, 0.015)
        theta_error = _wrap_to_pi(target_theta - float(self._pose.theta))
        return (
            abs(float(self._pen_x) - float(corner_x)) < corner_pos_tol_x
            and abs(float(self._pen_y) - float(corner_y)) < corner_pos_tol_y
            and abs(theta_error) < corner_theta_tol
        )

    def _line_tracking_along_scale(
        self,
        cross_track_error: float,
        theta_error: float,
    ) -> float:
        cross_abs = abs(cross_track_error)
        theta_abs = abs(theta_error)
        if cross_abs > 0.04 or theta_abs > 0.10:
            return 0.0
        cross_penalty = _clamp(cross_abs / 0.02, 0.0, 1.0)
        theta_penalty = _clamp(theta_abs / 0.05, 0.0, 1.0)
        penalty = max(cross_penalty, theta_penalty)
        return 1.0 - (0.45 * penalty)

    def _line_tracking_cmd(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        target_theta: float,
        speed_cap: float,
    ) -> Twist:
        k_cross = self._tick_params.k_y
        k_theta = self._tick_params.k_theta
        omega_sign = self._tick_params.omega_sign
        max_lat = self._tick_params.max_lateral_cmd
        max_ang = self._tick_params.max_angular_cmd
        axis = self._segment_axis(start_point, end_point)
        direction_sign = self._segment_direction_sign(start_point, end_point, axis)
        remaining_distance = self._remaining_distance(
            start_point,
            end_point,
            self._pen_x,
            self._pen_y,
        )
        cross_track_error = self._cross_track_error(
            start_point,
            end_point,
            self._pen_x,
            self._pen_y,
        )
        theta_error = _wrap_to_pi(target_theta - float(self._pose.theta))

        cmd = Twist()
        cmd.angular.z = _clamp(omega_sign * k_theta * theta_error, -max_ang, max_ang)

        if axis is None or direction_sign is None:
            return cmd

        along_cap = float(speed_cap)
        cross_cap = max_lat
        along_cmd = direction_sign * _clamp(2.0 * remaining_distance, -along_cap, along_cap)
        along_cmd *= self._line_tracking_along_scale(cross_track_error, theta_error)
        cross_cmd = _clamp(k_cross * cross_track_error, -cross_cap, cross_cap)
        cmd.linear.x, cmd.linear.y = self._segment_linear_components(
            axis,
            along_cmd,
            cross_cmd,
        )

        return cmd

    def _tracking_cmd(
        self,
        target_x: float,
        target_y: float,
        target_theta: float,
        speed_cap: float,
    ) -> Twist:
        k_y = self._tick_params.k_y
        k_theta = self._tick_params.k_theta
        omega_sign = self._tick_params.omega_sign
        max_lat = self._tick_params.max_lateral_cmd
        max_ang = self._tick_params.max_angular_cmd

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

    def _segment_complete(
        self,
        target_x: float,
        target_y: float,
        target_theta: float,
    ) -> bool:
        pos_tol_x = self._tick_params.pos_tol_x
        pos_tol_y = self._tick_params.pos_tol_y
        theta_tol = self._tick_params.theta_tol

        theta_error = _wrap_to_pi(target_theta - float(self._pose.theta))
        return (
            abs(float(self._pen_x) - target_x) < pos_tol_x
            and abs(float(self._pen_y) - target_y) < pos_tol_y
            and abs(theta_error) < theta_tol
        )

    def _enter_probe_state(self, pen_up_pos: float) -> None:
        self._publish_zero_twist()
        self._publish_pen(pen_up_pos)
        self._probe_target = pen_up_pos
        self._probe_cycle_counter = 0
        self._set_state(PEN_PROBE)

    def _start_pen_up_wait(self, next_state: str) -> None:
        self._pen_lift_state_start_sec = self._now_sec()
        self._next_state_after_pen_up = next_state
        self._set_state(PEN_UP)

    def _handle_pen_up_wait(self) -> None:
        pen_up_pos = self._tick_params.pen_up_pos
        pen_clear_gap = self._tick_params.pen_clear_gap
        pen_lift_timeout_sec = self._tick_params.pen_lift_timeout_sec
        pen_contact_timeout_sec = self._tick_params.pen_contact_timeout_sec

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

    def _probe_step(self) -> None:
        pen_down_max = self._tick_params.pen_down_max_pos
        pen_up_pos = self._tick_params.pen_up_pos
        probe_step = self._tick_params.pen_probe_step
        probe_period = self._tick_params.pen_probe_period_cycles
        contact_timeout_sec = self._tick_params.pen_contact_timeout_sec
        max_retries = self._tick_params.max_probe_retries_per_line
        min_probe_descent = max(3.0 * probe_step, 0.0015)
        min_contact_accept_target = _clamp(
            pen_up_pos - min_probe_descent,
            pen_down_max,
            pen_up_pos,
        )

        if not self._pen_data_fresh(contact_timeout_sec):
            self._fail_execution(
                'Pen contact/gap data stale during PEN_PROBE; stopping safely.'
            )
            return

        if self._probe_target is None:
            self._probe_target = pen_up_pos

        # Reject obviously spurious early contact while the probe is still
        # effectively at pen-up. This keeps probing collision-first, but avoids
        # accepting stale/false contact before the tool has descended enough to
        # make first contact physically plausible.
        if (
            self._effective_contact(contact_timeout_sec)
            and self._probe_target <= min_contact_accept_target
        ):
            self._draw_pen_target = _clamp(
                self._probe_target,
                pen_down_max,
                pen_up_pos,
            )
            self._probe_retries = 0
            self._lost_contact_cycles = 0
            self._set_state(PEN_SETTLE)
            return

        self._publish_zero_twist()

        self._probe_cycle_counter += 1
        if self._probe_cycle_counter >= probe_period:
            self._probe_target = max(pen_down_max, self._probe_target - probe_step)
            self._probe_cycle_counter = 0

        self._publish_pen(self._probe_target)

        if self._probe_target <= pen_down_max and not self._effective_contact(contact_timeout_sec):
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

    def _update_draw_pen_target(self, target: float) -> float:
        pen_down_max = self._tick_params.pen_down_max_pos
        pen_up_pos = self._tick_params.pen_up_pos
        # Collision-first contact keeps the current target shallow; legacy
        # gap-window recovery parameters remain compatibility no-ops.
        return _clamp(target, pen_down_max, pen_up_pos)

    def _build_debug_payload(self, transition_reason: str) -> dict[str, Any]:
        start_point, end_point = self._current_segment_points()
        axis = self._segment_axis(start_point, end_point)
        payload = {
            'state': self._state,
            'primitive_index': self._primitive_index,
            'segment_index': self._segment_index,
            'segment_axis': axis,
            'drawing_active': bool(self._drawing_active),
            'pen_contact': bool(self._pen_contact),
            'along_progress': None,
            'remaining_distance': None,
            'cross_track_error': None,
            'theta_error': None,
            'corner_residual_x': None,
            'corner_residual_y': None,
            'contact_lost_cycles': int(self._lost_contact_cycles),
            'transition_reason': transition_reason,
        }
        if self._pose is not None:
            payload['theta_error'] = _wrap_to_pi(
                self._tick_params.target_theta - float(self._pose.theta)
            )
        if (
            start_point is not None
            and end_point is not None
            and self._pen_x is not None
            and self._pen_y is not None
        ):
            payload['along_progress'] = self._along_track_progress(
                start_point,
                end_point,
                self._pen_x,
                self._pen_y,
            )
            payload['remaining_distance'] = self._remaining_distance(
                start_point,
                end_point,
                self._pen_x,
                self._pen_y,
            )
            payload['cross_track_error'] = self._cross_track_error(
                start_point,
                end_point,
                self._pen_x,
                self._pen_y,
            )
            payload['corner_residual_x'] = float(self._pen_x) - float(end_point[0])
            payload['corner_residual_y'] = float(self._pen_y) - float(end_point[1])
        return payload

    def _maybe_publish_debug(self) -> None:
        if not self._tick_params.publish_debug_telemetry:
            self._debug_transition_reason = None
            return
        self._debug_tick_counter += 1
        reason = self._debug_transition_reason
        if reason is None and (self._debug_tick_counter % 5) != 0:
            return
        if reason is None:
            reason = 'tick'
        msg = String()
        msg.data = json.dumps(
            self._build_debug_payload(reason),
            separators=(',', ':'),
        )
        self._debug_pub.publish(msg)
        self._debug_transition_reason = None

    def _fail_execution(self, reason: str) -> None:
        self.get_logger().warn(reason)
        self._publish_zero_twist()
        self._publish_pen(float(self.get_parameter('pen_up_pos').value))
        self._publish_drawing_active(False)
        self._set_status('error')
        self._set_state(DONE)

    def _handle_disabled(self) -> bool:
        if self._tick_params.enabled:
            return False
        if self._enabled_last and self._tick_params.publish_zero_on_stop:
            self._publish_zero_twist()
            self._publish_pen(self._tick_params.pen_up_pos)
        # Re-enable drawing for manual control when stroke executor is disabled.
        self._publish_drawing_active(True)
        self._reset_execution()
        self._set_status('idle')
        return True

    def _handle_enable_transition(self) -> None:
        if not (self._tick_params.enabled and not self._enabled_last):
            return
        self._reset_execution()
        if self._current_exec_path is not None and len(self._current_exec_path.primitives) > 0:
            self._set_status('running')
            self._set_state(MOVE_TO_STROKE_START)
            self.get_logger().info('enabled=true, starting loaded stroke plan.')
            return
        self.get_logger().info('enabled=true, waiting for a valid stroke plan.')
        self._set_status('idle')

    def _handle_unavailable_pose_or_board(self) -> bool:
        if self._board is not None and self._pose_fresh(self._tick_params.pose_timeout_sec):
            return False
        if self._current_exec_path is not None and self._state not in (IDLE, DONE):
            self._fail_execution(
                'Robot pose/board data stale during stroke execution; stopping safely.'
            )
        else:
            self._publish_zero_twist()
            self._publish_pen(self._tick_params.pen_up_pos)
            self._set_status('idle')
        return True

    def _handle_unavailable_pen_pose(self) -> bool:
        if self._pen_pose_fresh(self._tick_params.pen_pose_timeout_sec):
            return False
        if self._current_exec_path is not None and self._state not in (IDLE, DONE):
            self._fail_execution('Pen pose stale during stroke execution; stopping safely.')
        else:
            self._publish_zero_twist()
            self._publish_pen(self._tick_params.pen_up_pos)
            self._set_status('idle')
        return True

    def _handle_missing_exec_path(self) -> bool:
        if self._current_exec_path is not None:
            return False
        self._publish_zero_twist()
        self._publish_pen(self._tick_params.pen_up_pos)
        self._set_status('idle')
        return True

    def _handle_state_idle(self) -> None:
        self._publish_zero_twist()
        self._publish_pen(self._tick_params.pen_up_pos)

    def _handle_state_move_to_stroke_start(self) -> None:
        primitive = self._current_primitive()
        if primitive is None:
            self._set_state(DONE)
            return
        start_point, _ = self._current_segment_points()
        if start_point is None:
            self._set_state(ADVANCE_STROKE)
            return
        self._publish_pen(self._tick_params.pen_up_pos)
        self._publish_drawing_active(False)
        cmd = self._tracking_cmd(
            start_point[0],
            start_point[1],
            self._tick_params.target_theta,
            self._tick_params.reposition_speed,
        )
        self._cmd_pub.publish(cmd)
        if self._segment_complete(
            start_point[0],
            start_point[1],
            self._tick_params.target_theta,
        ):
            self._probe_retries = 0
            self._lost_contact_cycles = 0
            self._draw_pen_target = None
            if self._current_primitive_is_drawn():
                self._enter_probe_state(self._tick_params.pen_up_pos)
            else:
                self._set_state(DRAW_SEGMENT)

    def _handle_state_pen_probe(self) -> None:
        self._publish_drawing_active(False)
        self._probe_step()

    def _handle_state_pen_settle(self) -> bool:
        if not self._pen_data_fresh(self._tick_params.pen_contact_timeout_sec):
            self._fail_execution(
                'Pen contact/gap data stale during PEN_SETTLE; stopping safely.'
            )
            return True
        self._publish_zero_twist()
        self._publish_drawing_active(False)
        if self._draw_pen_target is None:
            self._draw_pen_target = self._tick_params.pen_down_min_pos
        self._publish_pen(self._draw_pen_target)
        self._pen_settle_counter += 1
        if self._pen_settle_counter >= self._tick_params.pen_settle_cycles:
            self._set_state(DRAW_SEGMENT)
        return False

    def _handle_state_draw_segment(self) -> bool:
        primitive = self._current_primitive()
        start_point, end_point = self._current_segment_points()
        if primitive is None or start_point is None or end_point is None:
            self._set_state(ADVANCE_STROKE)
            return False

        if self._current_primitive_is_drawn():
            if not self._pen_data_fresh(self._tick_params.pen_contact_timeout_sec):
                self._fail_execution(
                    'Pen contact/gap data stale during DRAW_SEGMENT; stopping safely.'
                )
                return True
            if self._tick_params.contact_required_for_drawing:
                if self._effective_contact(self._tick_params.pen_contact_timeout_sec):
                    self._lost_contact_cycles = 0
                else:
                    self._lost_contact_cycles += 1
                    if (
                        self._lost_contact_cycles
                        >= self._tick_params.lost_contact_cycles_before_reprobe
                    ):
                        self.get_logger().warn(
                            'Contact lost during DRAW_SEGMENT, re-probing.'
                        )
                        self._enter_probe_state(self._tick_params.pen_up_pos)
                        return False

            if self._draw_pen_target is None:
                self._draw_pen_target = self._tick_params.pen_down_min_pos
            self._draw_pen_target = self._update_draw_pen_target(self._draw_pen_target)
            self._publish_pen(self._draw_pen_target)

            required_draw_delay_cycles = (
                self._tick_params.draw_start_delay_cycles
                + self._post_corner_draw_delay_cycles
            )
            if self._draw_segment_cycles >= required_draw_delay_cycles:
                self._publish_drawing_active(True)
            else:
                self._publish_drawing_active(False)
            self._draw_segment_cycles += 1
            speed_cap = self._tick_params.draw_speed
        else:
            self._publish_pen(self._tick_params.pen_up_pos)
            self._publish_drawing_active(False)
            speed_cap = self._tick_params.reposition_speed

        if self._current_primitive_is_drawn():
            cmd = self._line_tracking_cmd(
                start_point,
                end_point,
                self._tick_params.target_theta,
                speed_cap,
            )
        else:
            cmd = self._tracking_cmd(
                end_point[0],
                end_point[1],
                self._tick_params.target_theta,
                speed_cap,
            )
        self._cmd_pub.publish(cmd)

        if self._current_primitive_is_drawn():
            segment_reached = self._segment_ready(
                start_point,
                end_point,
                self._tick_params.target_theta,
            )
        else:
            segment_reached = self._segment_complete(
                end_point[0],
                end_point[1],
                self._tick_params.target_theta,
            )
        if not segment_reached:
            return False

        if self._current_segment_is_last():
            if self._current_primitive_is_drawn():
                self._debug_transition_reason = 'segment_end:pen_up'
                self._start_pen_up_wait(ADVANCE_STROKE)
            else:
                self._debug_transition_reason = 'segment_end:advance_stroke'
                self._set_state(ADVANCE_STROKE)
            return False

        self._debug_transition_reason = 'segment_end:corner_settle'
        self._set_state(CORNER_SETTLE)
        return False

    def _handle_state_corner_settle(self) -> None:
        self._publish_drawing_active(False)
        if self._current_transition_keeps_pen_down() and self._draw_pen_target is not None:
            self._publish_pen(self._draw_pen_target)

        _, corner_point = self._current_segment_points()
        if corner_point is None:
            self._publish_zero_twist()
            self._corner_stable_counter = 0
        else:
            cmd = self._tracking_cmd(
                corner_point[0],
                corner_point[1],
                self._tick_params.target_theta,
                min(self._tick_params.draw_speed, self._tick_params.reposition_speed),
            )
            self._cmd_pub.publish(cmd)
            if self._current_transition_keeps_pen_down() and self._corner_hold_ready(
                corner_point[0],
                corner_point[1],
                self._tick_params.target_theta,
            ):
                self._corner_stable_counter += 1
            else:
                self._corner_stable_counter = 0

        self._corner_settle_counter += 1
        corner_ready_to_advance = (
            not self._current_transition_keeps_pen_down()
            or self._corner_stable_counter >= 2
        )
        corner_timeout_reached = self._corner_settle_counter >= max(
            self._tick_params.corner_settle_cycles + 8,
            12,
        )
        if not (
            (
                self._corner_settle_counter >= self._tick_params.corner_settle_cycles
                and corner_ready_to_advance
            )
            or corner_timeout_reached
        ):
            return

        self._post_corner_draw_delay_cycles = (
            2 if self._current_transition_keeps_pen_down() else 0
        )
        if corner_timeout_reached and not corner_ready_to_advance:
            self._debug_transition_reason = 'corner_settle:timeout'
            self.get_logger().warn(
                'Corner settle timed out before exact tool-tip stabilization; '
                'continuing to the next segment.'
            )
        else:
            self._debug_transition_reason = 'corner_settle:ready'
        self._set_state(ADVANCE_SEGMENT)

    def _handle_state_pen_up(self) -> bool:
        if not self._pen_data_fresh(self._tick_params.pen_contact_timeout_sec):
            self._fail_execution(
                'Pen contact/gap data stale during PEN_UP verification; stopping safely.'
            )
            return True
        self._handle_pen_up_wait()
        return False

    def _handle_state_advance_segment(self) -> None:
        primitive = self._current_primitive()
        points = self._current_primitive_points()
        if primitive is None:
            self._set_state(ADVANCE_STROKE)
            return
        self._segment_index += 1
        if self._segment_index >= len(points) - 1:
            self._debug_transition_reason = 'advance_segment:advance_stroke'
            self._set_state(ADVANCE_STROKE)
            return
        self._lost_contact_cycles = 0
        self._debug_transition_reason = 'advance_segment:draw_segment'
        self._set_state(DRAW_SEGMENT)

    def _handle_state_advance_stroke(self) -> None:
        self._primitive_index += 1
        self._segment_index = 0
        self._probe_retries = 0
        self._lost_contact_cycles = 0
        self._draw_pen_target = None
        if self._current_exec_path is None:
            self._set_state(DONE)
        elif self._primitive_index >= len(self._current_exec_path.primitives):
            self._set_status('done')
            self._set_state(DONE)
        else:
            self._set_state(MOVE_TO_STROKE_START)

    def _handle_state_done(self) -> None:
        self._publish_zero_twist()
        self._publish_pen(self._tick_params.pen_up_pos)
        self._publish_drawing_active(False)

    def _dispatch_state(self) -> bool:
        if self._state == IDLE:
            self._handle_state_idle()
            return False
        if self._state == MOVE_TO_STROKE_START:
            self._handle_state_move_to_stroke_start()
            return False
        if self._state == PEN_PROBE:
            self._handle_state_pen_probe()
            return False
        if self._state == PEN_SETTLE:
            return self._handle_state_pen_settle()
        if self._state == DRAW_SEGMENT:
            return self._handle_state_draw_segment()
        if self._state == CORNER_SETTLE:
            self._handle_state_corner_settle()
            return False
        if self._state == PEN_UP:
            return self._handle_state_pen_up()
        if self._state == ADVANCE_SEGMENT:
            self._handle_state_advance_segment()
            return False
        if self._state == ADVANCE_STROKE:
            self._handle_state_advance_stroke()
            return False
        if self._state == DONE:
            self._handle_state_done()
            return False
        return False

    def _on_timer(self) -> None:
        self._tick_params = self._read_tick_params()

        if self._pending_external_plan is not None and self._board is not None:
            self._finalize_pending_plan()

        if self._handle_disabled():
            self._enabled_last = False
            return

        self._handle_enable_transition()

        if self._handle_unavailable_pose_or_board():
            self._enabled_last = True
            return
        if self._handle_unavailable_pen_pose():
            self._enabled_last = True
            return
        if self._handle_missing_exec_path():
            self._enabled_last = True
            return

        if self._dispatch_state():
            self._enabled_last = True
            return

        self._maybe_publish_debug()
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
