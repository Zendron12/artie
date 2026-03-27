from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String

from wall_climber.arm_writer_kinematics import ArmGeometry, IKConfig, solve_best_ik


TIMER_HZ = 60.0
IDLE = 'IDLE'
MOVE_TO_START = 'MOVE_TO_START'
PEN_PROBE = 'PEN_PROBE'
PEN_SETTLE = 'PEN_SETTLE'
DRAW_SEGMENT = 'DRAW_SEGMENT'
ADVANCE_SEGMENT = 'ADVANCE_SEGMENT'
PEN_UP = 'PEN_UP'
DONE = 'DONE'
ERROR = 'ERROR'
ACTIVE_STATES = {
    MOVE_TO_START,
    PEN_PROBE,
    PEN_SETTLE,
    DRAW_SEGMENT,
    ADVANCE_SEGMENT,
    PEN_UP,
}


@dataclass(frozen=True)
class ArmWriterParams:
    enabled: bool
    draw_speed: float
    travel_speed: float
    position_tolerance: float
    contact_required_for_drawing: bool
    contact_gap_min: float
    contact_gap_max: float
    pen_up_pos: float
    pen_down_min_pos: float
    pen_down_max_pos: float
    pen_probe_step: float
    pen_settle_cycles: int
    pen_contact_timeout_sec: float
    pen_pose_timeout_sec: float
    pose_timeout_sec: float
    pen_lift_timeout_sec: float
    body_drift_pos_tol: float
    body_drift_theta_tol: float
    arm_mount_y: float
    local_x_min: float
    local_x_max: float
    local_y_min: float
    local_y_max: float
    reachability_sample_spacing: float
    shoulder_limit_margin: float
    ik_max_iterations: int
    ik_damping: float
    ik_convergence_tol: float
    ik_final_error_tol: float
    ik_finite_diff_eps: float
    ik_max_step: float


@dataclass(frozen=True)
class PathPrimitive:
    draw: bool
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ExecutionPlan:
    primitives: tuple[PathPrimitive, ...]


@dataclass(frozen=True)
class AnchorPose:
    robot_x: float
    robot_y: float
    theta: float
    shoulder_x: float
    shoulder_y: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ArmWriterController(Node):
    def __init__(self) -> None:
        super().__init__('arm_writer_controller')

        self.declare_parameter('enabled', False)
        self.declare_parameter('draw_speed', 0.06)
        self.declare_parameter('travel_speed', 0.10)
        self.declare_parameter('position_tolerance', 0.004)
        self.declare_parameter('contact_required_for_drawing', True)
        self.declare_parameter('contact_gap_min', -0.0018)
        self.declare_parameter('contact_gap_max', 0.0018)
        self.declare_parameter('pen_up_pos', 0.018)
        self.declare_parameter('pen_down_min_pos', -0.010)
        self.declare_parameter('pen_down_max_pos', -0.030)
        self.declare_parameter('pen_probe_step', 0.0005)
        self.declare_parameter('pen_settle_cycles', 12)
        self.declare_parameter('pen_contact_timeout_sec', 1.5)
        self.declare_parameter('pen_pose_timeout_sec', 0.5)
        self.declare_parameter('pose_timeout_sec', 0.5)
        self.declare_parameter('pen_lift_timeout_sec', 1.5)
        self.declare_parameter('body_drift_pos_tol', 0.005)
        self.declare_parameter('body_drift_theta_tol', 0.035)
        self.declare_parameter('arm_mount_y', 0.10)
        self.declare_parameter('local_x_min', -0.09)
        self.declare_parameter('local_x_max', 0.09)
        self.declare_parameter('local_y_min', 0.24)
        self.declare_parameter('local_y_max', 0.31)
        self.declare_parameter('reachability_sample_spacing', 0.005)
        self.declare_parameter('shoulder_limit_margin', 0.12)
        self.declare_parameter('ik_max_iterations', 12)
        self.declare_parameter('ik_damping', 1.0e-3)
        self.declare_parameter('ik_convergence_tol', 1.0e-4)
        self.declare_parameter('ik_final_error_tol', 0.002)
        self.declare_parameter('ik_finite_diff_eps', 1.0e-4)
        self.declare_parameter('ik_max_step', 0.25)

        self._geometry = ArmGeometry()

        self._pose: Pose2D | None = None
        self._pose_stamp = None
        self._pen_xy: tuple[float, float] | None = None
        self._pen_pose_stamp = None
        self._pen_contact = False
        self._pen_contact_stamp = None
        self._pen_gap = float('nan')
        self._pen_gap_stamp = None
        self._board: dict[str, Any] | None = None
        self._stroke_executor_status = 'idle'

        self._status = 'idle'
        self._state = IDLE
        self._enabled_last = False
        self._drawing_active = False

        self._pending_plan: dict[str, Any] | None = None
        self._current_plan: ExecutionPlan | None = None
        self._primitive_index = 0
        self._segment_index = 0

        self._anchor: AnchorPose | None = None
        self._current_shoulder_targets: tuple[float, float] | None = None
        self._last_ik_solution: tuple[float, float] | None = None
        self._probe_target: float | None = None
        self._draw_pen_target: float | None = None
        self._pen_settle_counter = 0
        self._pen_lift_started_sec: float | None = None
        self._next_state_after_pen_up: str | None = None

        self.create_subscription(
            String, '/wall_climber/arm_stroke_plan', self._plan_cb, 10
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
        self.create_subscription(
            String,
            '/wall_climber/stroke_executor_status',
            self._stroke_executor_status_cb,
            10,
        )

        self._arm_pub = self.create_publisher(
            JointState, '/wall_climber/arm_joint_targets', 10
        )
        self._pen_pub = self.create_publisher(
            Float64, '/wall_climber/pen_target', 10
        )
        self._drawing_active_pub = self.create_publisher(
            Bool, '/wall_climber/drawing_active', 10
        )
        self._status_pub = self.create_publisher(
            String, '/wall_climber/arm_writer_status', 10
        )

        self._timer = self.create_timer(1.0 / TIMER_HZ, self._on_timer)
        self._set_status('idle')
        self.get_logger().info('Arm writer controller ready (enabled=false).')

    def _read_params(self) -> ArmWriterParams:
        return ArmWriterParams(
            enabled=bool(self.get_parameter('enabled').value),
            draw_speed=float(self.get_parameter('draw_speed').value),
            travel_speed=float(self.get_parameter('travel_speed').value),
            position_tolerance=float(
                self.get_parameter('position_tolerance').value
            ),
            contact_required_for_drawing=bool(
                self.get_parameter('contact_required_for_drawing').value
            ),
            contact_gap_min=float(self.get_parameter('contact_gap_min').value),
            contact_gap_max=float(self.get_parameter('contact_gap_max').value),
            pen_up_pos=float(self.get_parameter('pen_up_pos').value),
            pen_down_min_pos=float(self.get_parameter('pen_down_min_pos').value),
            pen_down_max_pos=float(self.get_parameter('pen_down_max_pos').value),
            pen_probe_step=float(self.get_parameter('pen_probe_step').value),
            pen_settle_cycles=max(
                1, int(self.get_parameter('pen_settle_cycles').value)
            ),
            pen_contact_timeout_sec=float(
                self.get_parameter('pen_contact_timeout_sec').value
            ),
            pen_pose_timeout_sec=float(
                self.get_parameter('pen_pose_timeout_sec').value
            ),
            pose_timeout_sec=float(self.get_parameter('pose_timeout_sec').value),
            pen_lift_timeout_sec=float(
                self.get_parameter('pen_lift_timeout_sec').value
            ),
            body_drift_pos_tol=float(
                self.get_parameter('body_drift_pos_tol').value
            ),
            body_drift_theta_tol=float(
                self.get_parameter('body_drift_theta_tol').value
            ),
            arm_mount_y=float(self.get_parameter('arm_mount_y').value),
            local_x_min=float(self.get_parameter('local_x_min').value),
            local_x_max=float(self.get_parameter('local_x_max').value),
            local_y_min=float(self.get_parameter('local_y_min').value),
            local_y_max=float(self.get_parameter('local_y_max').value),
            reachability_sample_spacing=max(
                1.0e-4,
                float(self.get_parameter('reachability_sample_spacing').value),
            ),
            shoulder_limit_margin=float(
                self.get_parameter('shoulder_limit_margin').value
            ),
            ik_max_iterations=max(
                1, int(self.get_parameter('ik_max_iterations').value)
            ),
            ik_damping=float(self.get_parameter('ik_damping').value),
            ik_convergence_tol=float(
                self.get_parameter('ik_convergence_tol').value
            ),
            ik_final_error_tol=float(
                self.get_parameter('ik_final_error_tol').value
            ),
            ik_finite_diff_eps=float(
                self.get_parameter('ik_finite_diff_eps').value
            ),
            ik_max_step=float(self.get_parameter('ik_max_step').value),
        )

    def _ik_config(self, params: ArmWriterParams) -> IKConfig:
        return IKConfig(
            max_iterations=params.ik_max_iterations,
            damping=params.ik_damping,
            finite_diff_eps=params.ik_finite_diff_eps,
            convergence_tol=params.ik_convergence_tol,
            final_error_tol=params.ik_final_error_tol,
            shoulder_limit_margin=params.shoulder_limit_margin,
            max_step=params.ik_max_step,
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose_cb(self, msg: Pose2D) -> None:
        self._pose = msg
        self._pose_stamp = self.get_clock().now()

    def _pen_pose_cb(self, msg: PointStamped) -> None:
        self._pen_xy = (float(msg.point.x), float(msg.point.y))
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
            needed = (
                'writable_x_min',
                'writable_x_max',
                'writable_y_min',
                'writable_y_max',
            )
            if any(key not in data for key in needed):
                self.get_logger().warn(
                    'board_info JSON missing writable bounds keys.'
                )
                return
            self._board = data
        except Exception as exc:
            self.get_logger().warn(f'Failed to parse /wall_climber/board_info: {exc}')

    def _stroke_executor_status_cb(self, msg: String) -> None:
        self._stroke_executor_status = str(msg.data).strip()

    def _set_status(self, text: str) -> None:
        if self._status == text:
            return
        self._status = text
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)

    def _publish_pen(self, value: float) -> None:
        msg = Float64()
        msg.data = float(value)
        self._pen_pub.publish(msg)

    def _publish_drawing_active(self, active: bool) -> None:
        active = bool(active)
        if self._drawing_active == active:
            return
        self._drawing_active = active
        msg = Bool()
        msg.data = active
        self._drawing_active_pub.publish(msg)

    def _publish_arm_target(self, theta_l: float, theta_r: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['left_shoulder_joint', 'right_shoulder_joint']
        msg.position = [float(theta_l), float(theta_r)]
        self._arm_pub.publish(msg)
        self._current_shoulder_targets = (float(theta_l), float(theta_r))
        self._last_ik_solution = self._current_shoulder_targets

    def _hold_arm_targets(self) -> None:
        if self._current_shoulder_targets is None:
            return
        self._publish_arm_target(*self._current_shoulder_targets)

    def _normalize_plan(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(payload, dict):
            return None, 'Arm stroke plan must be a JSON object.'

        frame = payload.get('frame')
        if frame != 'board':
            return None, 'arm_stroke_plan.frame must be exactly "board".'

        strokes = payload.get('strokes')
        if not isinstance(strokes, list):
            return None, 'arm_stroke_plan.strokes must be a list.'

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
                return None, (
                    f'stroke[{index}] of type line must contain exactly 2 points.'
                )
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

    def _build_execution_plan(self, external_plan: dict[str, Any]) -> ExecutionPlan:
        primitives = []
        for stroke in external_plan['strokes']:
            primitives.append(
                PathPrimitive(
                    draw=bool(stroke['draw']),
                    points=tuple(stroke['points']),
                )
            )
        return ExecutionPlan(primitives=tuple(primitives))

    def _plan_cb(self, msg: String) -> None:
        if self._state in ACTIVE_STATES:
            self.get_logger().warn('Arm writer is busy; rejecting new arm_stroke_plan.')
            return
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self._reject_plan(f'Malformed arm stroke plan JSON: {exc}')
            return

        normalized, error = self._normalize_plan(payload)
        if error is not None:
            self._reject_plan(error)
            return

        self._pending_plan = normalized
        if self._board is None:
            self.get_logger().warn(
                'Received structurally valid arm stroke plan, but board_info is not ready yet; '
                'deferring validation.'
            )

    def _reject_plan(self, reason: str) -> None:
        self.get_logger().warn(reason)
        self._pending_plan = None
        self._current_plan = None
        self._publish_pen(float(self.get_parameter('pen_up_pos').value))
        self._publish_drawing_active(False)
        self._set_status('error')
        self._state = ERROR
        self._next_state_after_pen_up = None
        self._pen_lift_started_sec = None

    def _point_inside_writable(self, point: tuple[float, float]) -> bool:
        if self._board is None:
            return False
        x, y = point
        return (
            float(self._board['writable_x_min']) <= x <= float(self._board['writable_x_max'])
            and float(self._board['writable_y_min']) <= y <= float(self._board['writable_y_max'])
        )

    def _segment_is_axis_aligned(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        epsilon: float = 1.0e-6,
    ) -> bool:
        dx = abs(float(end_point[0]) - float(start_point[0]))
        dy = abs(float(end_point[1]) - float(start_point[1]))
        return dx <= epsilon or dy <= epsilon

    def _validate_plan_points(self, plan: dict[str, Any]) -> str | None:
        if self._board is None:
            return 'board_info is not available; cannot validate arm stroke points yet.'
        for stroke_index, stroke in enumerate(plan['strokes']):
            for point_index, point in enumerate(stroke['points']):
                if not self._point_inside_writable(point):
                    return (
                        'Arm stroke plan rejected: '
                        f'stroke[{stroke_index}].points[{point_index}]={point} '
                        'is outside writable board bounds.'
                    )
            for segment_index in range(len(stroke['points']) - 1):
                start_point = stroke['points'][segment_index]
                end_point = stroke['points'][segment_index + 1]
                if not self._segment_is_axis_aligned(start_point, end_point):
                    return (
                        'Arm stroke plan rejected: '
                        f'stroke[{stroke_index}] segment[{segment_index}] '
                        f'from {start_point} to {end_point} is not axis-aligned. '
                        'Arm Writer V1 supports only horizontal/vertical segments.'
                    )
        return None

    def _pose_fresh(self, timeout_sec: float) -> bool:
        if self._pose is None or self._pose_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pose_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_pose_fresh(self, timeout_sec: float) -> bool:
        if self._pen_xy is None or self._pen_pose_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pen_pose_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_contact_fresh(self, timeout_sec: float) -> bool:
        if self._pen_contact_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pen_contact_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_gap_fresh(self, timeout_sec: float) -> bool:
        if self._pen_gap_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pen_gap_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_data_fresh(self, timeout_sec: float) -> bool:
        return self._pen_contact_fresh(timeout_sec) and self._pen_gap_fresh(
            timeout_sec
        )

    def _capture_anchor(self, params: ArmWriterParams) -> AnchorPose | None:
        if self._pose is None:
            return None
        theta = float(self._pose.theta)
        shoulder_dx = params.arm_mount_y * math.sin(theta)
        shoulder_dy = -params.arm_mount_y * math.cos(theta)
        return AnchorPose(
            robot_x=float(self._pose.x),
            robot_y=float(self._pose.y),
            theta=theta,
            shoulder_x=float(self._pose.x) + shoulder_dx,
            shoulder_y=float(self._pose.y) + shoulder_dy,
        )

    def _board_to_local(
        self,
        board_point: tuple[float, float],
        anchor: AnchorPose,
    ) -> tuple[float, float]:
        dx = float(board_point[0]) - anchor.shoulder_x
        dy = float(board_point[1]) - anchor.shoulder_y
        local_x = dx * math.cos(anchor.theta) + dy * math.sin(anchor.theta)
        local_y = dx * math.sin(anchor.theta) - dy * math.cos(anchor.theta)
        return local_x, local_y

    def _local_point_is_safe(
        self,
        local_point: tuple[float, float],
        params: ArmWriterParams,
    ) -> bool:
        return (
            params.local_x_min <= local_point[0] <= params.local_x_max
            and params.local_y_min <= local_point[1] <= params.local_y_max
        )

    def _sample_segment_points(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        spacing: float,
    ) -> list[tuple[float, float]]:
        points = [start_point]
        dist = math.hypot(
            end_point[0] - start_point[0],
            end_point[1] - start_point[1],
        )
        if dist > spacing:
            subdivisions = int(math.ceil(dist / spacing))
            for idx in range(1, subdivisions):
                ratio = idx / subdivisions
                points.append(
                    (
                        start_point[0] + (end_point[0] - start_point[0]) * ratio,
                        start_point[1] + (end_point[1] - start_point[1]) * ratio,
                    )
                )
        if end_point != start_point:
            points.append(end_point)
        return points

    def _ik_seeds(
        self,
        extra_seed: tuple[float, float] | None = None,
    ) -> list[tuple[float, float]]:
        seeds: list[tuple[float, float]] = []
        if extra_seed is not None:
            seeds.append(extra_seed)
        if self._current_shoulder_targets is not None:
            seeds.append(self._current_shoulder_targets)
        if self._last_ik_solution is not None:
            seeds.append(self._last_ik_solution)
        seeds.append((0.0, 0.0))
        return seeds

    def _solve_local_target(
        self,
        local_target: tuple[float, float],
        params: ArmWriterParams,
        extra_seed: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        result = solve_best_ik(
            local_target,
            self._ik_seeds(extra_seed),
            geom=self._geometry,
            config=self._ik_config(params),
        )
        if result is None:
            return None
        return result.theta_l, result.theta_r

    def _validate_reachability(
        self,
        plan: dict[str, Any],
        anchor: AnchorPose,
        params: ArmWriterParams,
    ) -> str | None:
        seed = self._last_ik_solution
        for stroke_index, stroke in enumerate(plan['strokes']):
            points = stroke['points']
            for segment_index in range(len(points) - 1):
                sampled = self._sample_segment_points(
                    points[segment_index],
                    points[segment_index + 1],
                    params.reachability_sample_spacing,
                )
                for sample_index, point in enumerate(sampled):
                    local_point = self._board_to_local(point, anchor)
                    if not self._local_point_is_safe(local_point, params):
                        return (
                            'Arm stroke plan rejected: '
                            f'stroke[{stroke_index}] segment[{segment_index}] '
                            f'sample[{sample_index}]={point} maps to local '
                            f'{local_point}, outside safe local workspace.'
                        )
                    solved = self._solve_local_target(
                        local_point,
                        params,
                        extra_seed=seed,
                    )
                    if solved is None:
                        return (
                            'Arm stroke plan rejected: '
                            f'stroke[{stroke_index}] segment[{segment_index}] '
                            f'sample[{sample_index}]={point} is not safely '
                            'reachable for Arm Writer V1.'
                        )
                    seed = solved
        return None

    def _reset_execution(self) -> None:
        self._current_plan = None
        self._primitive_index = 0
        self._segment_index = 0
        self._anchor = None
        self._probe_target = None
        self._draw_pen_target = None
        self._pen_settle_counter = 0
        self._pen_lift_started_sec = None
        self._next_state_after_pen_up = None

    def _maybe_start_pending_plan(self, params: ArmWriterParams) -> None:
        if self._pending_plan is None or not params.enabled:
            return
        if self._state not in {IDLE, DONE, ERROR}:
            return
        if self._stroke_executor_status == 'running':
            self._reject_plan(
                'Arm stroke plan rejected: stroke_executor_status=running.'
            )
            return
        if not self._pose_fresh(params.pose_timeout_sec):
            return
        if self._board is None:
            return

        error = self._validate_plan_points(self._pending_plan)
        if error is not None:
            self._reject_plan(error)
            return

        anchor = self._capture_anchor(params)
        if anchor is None:
            return

        error = self._validate_reachability(self._pending_plan, anchor, params)
        if error is not None:
            self._reject_plan(error)
            return

        self._current_plan = self._build_execution_plan(self._pending_plan)
        self._pending_plan = None
        self._primitive_index = 0
        self._segment_index = 0
        self._anchor = anchor
        self._probe_target = None
        self._draw_pen_target = None
        self._pen_settle_counter = 0
        self._pen_lift_started_sec = None
        self._next_state_after_pen_up = None
        self._publish_drawing_active(False)
        self._publish_pen(params.pen_up_pos)
        self._state = MOVE_TO_START
        self._set_status('running')
        self.get_logger().info(
            'Arm stroke plan accepted; starting arm-only execution.'
        )

    def _current_primitive(self) -> PathPrimitive | None:
        if self._current_plan is None:
            return None
        if not (0 <= self._primitive_index < len(self._current_plan.primitives)):
            return None
        return self._current_plan.primitives[self._primitive_index]

    def _current_segment_points(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        primitive = self._current_primitive()
        if primitive is None:
            return None
        if not (0 <= self._segment_index < len(primitive.points) - 1):
            return None
        return primitive.points[self._segment_index], primitive.points[
            self._segment_index + 1
        ]

    def _body_drift_exceeded(self, params: ArmWriterParams) -> bool:
        if self._anchor is None or self._pose is None:
            return True
        pos_drift = math.hypot(
            float(self._pose.x) - self._anchor.robot_x,
            float(self._pose.y) - self._anchor.robot_y,
        )
        theta_drift = abs(_wrap_to_pi(float(self._pose.theta) - self._anchor.theta))
        return (
            pos_drift > params.body_drift_pos_tol
            or theta_drift > params.body_drift_theta_tol
        )

    def _move_pen_toward(
        self,
        destination: tuple[float, float],
        speed: float,
        params: ArmWriterParams,
    ) -> tuple[bool, str | None]:
        if self._anchor is None:
            return False, 'Arm writer lost its body anchor.'
        if not self._pen_pose_fresh(params.pen_pose_timeout_sec):
            return False, 'Pen pose is stale during arm writing.'
        if self._pen_xy is None:
            return False, 'Pen pose is unavailable during arm writing.'

        current_x, current_y = self._pen_xy
        dx = destination[0] - current_x
        dy = destination[1] - current_y
        remaining = math.hypot(dx, dy)
        if remaining <= params.position_tolerance:
            return True, None

        step_len = min(speed / TIMER_HZ, remaining)
        if remaining > 1.0e-9:
            ratio = step_len / remaining
            intermediate = (current_x + dx * ratio, current_y + dy * ratio)
        else:
            intermediate = destination

        local_target = self._board_to_local(intermediate, self._anchor)
        if not self._local_point_is_safe(local_target, params):
            current_local = self._board_to_local((current_x, current_y), self._anchor)
            if not self._local_point_is_safe(current_local, params):
                local_target = (
                    _clamp(local_target[0], params.local_x_min, params.local_x_max),
                    _clamp(local_target[1], params.local_y_min, params.local_y_max),
                )
            else:
                return (
                    False,
                    f'Local target {local_target} is outside the safe arm workspace.',
                )

        solved = self._solve_local_target(local_target, params)
        if solved is None:
            return False, f'Failed to solve IK for local target {local_target}.'

        self._publish_arm_target(*solved)
        return False, None

    def _probe_min_accept_target(self, params: ArmWriterParams) -> float:
        min_probe_descent = max(3.0 * params.pen_probe_step, 0.0015)
        return _clamp(
            params.pen_up_pos - min_probe_descent,
            params.pen_down_max_pos,
            params.pen_up_pos,
        )

    def _effective_draw_contact(self, params: ArmWriterParams) -> bool:
        if not self._pen_data_fresh(params.pen_contact_timeout_sec):
            return False
        if not self._pen_contact:
            return False
        return params.contact_gap_min <= self._pen_gap <= params.contact_gap_max

    def _finish_current_primitive(self, params: ArmWriterParams) -> None:
        assert self._current_plan is not None
        self._segment_index = 0
        self._primitive_index += 1
        if self._primitive_index >= len(self._current_plan.primitives):
            self._next_state_after_pen_up = DONE
            self._state = PEN_UP
            self._pen_lift_started_sec = None
            return

        previous = self._current_plan.primitives[self._primitive_index - 1]
        if previous.draw:
            self._next_state_after_pen_up = MOVE_TO_START
            self._state = PEN_UP
            self._pen_lift_started_sec = None
        else:
            self._state = MOVE_TO_START
            self._publish_drawing_active(False)
            self._publish_pen(params.pen_up_pos)

    def _fail_active_execution(self, reason: str, params: ArmWriterParams) -> None:
        self.get_logger().warn(reason)
        self._publish_drawing_active(False)
        self._publish_pen(params.pen_up_pos)
        self._set_status('error')
        self._state = ERROR
        self._next_state_after_pen_up = None
        self._pen_lift_started_sec = None

    def _handle_move_to_start(self, params: ArmWriterParams) -> None:
        self._publish_drawing_active(False)
        self._publish_pen(params.pen_up_pos)
        segment = self._current_segment_points()
        if segment is None:
            self._fail_active_execution(
                'Arm writer lost the current segment during MOVE_TO_START.',
                params,
            )
            return

        reached, error = self._move_pen_toward(
            segment[0], params.travel_speed, params
        )
        if error is not None:
            self._fail_active_execution(error, params)
            return
        if reached:
            primitive = self._current_primitive()
            if primitive is None:
                self._fail_active_execution(
                    'Arm writer lost the current primitive at stroke start.',
                    params,
                )
                return
            if primitive.draw:
                self._probe_target = params.pen_up_pos
                self._draw_pen_target = None
                self._state = PEN_PROBE
            else:
                self._state = DRAW_SEGMENT

    def _handle_pen_probe(self, params: ArmWriterParams) -> None:
        self._publish_drawing_active(False)
        self._hold_arm_targets()
        if not self._pen_data_fresh(params.pen_contact_timeout_sec):
            self._fail_active_execution(
                'Pen contact/gap data are stale during PEN_PROBE.',
                params,
            )
            return
        if self._probe_target is None:
            self._probe_target = params.pen_up_pos

        min_accept_target = self._probe_min_accept_target(params)
        if self._pen_contact and self._probe_target <= min_accept_target:
            self._draw_pen_target = _clamp(
                self._probe_target,
                params.pen_down_max_pos,
                params.pen_down_min_pos,
            )
            self._pen_settle_counter = 0
            self._state = PEN_SETTLE
            return

        next_target = self._probe_target - params.pen_probe_step
        if next_target < params.pen_down_max_pos:
            self._fail_active_execution(
                'Pen probe failed: no contact before pen_down_max_pos.',
                params,
            )
            return
        self._probe_target = next_target
        self._publish_pen(self._probe_target)

    def _handle_pen_settle(self, params: ArmWriterParams) -> None:
        self._publish_drawing_active(False)
        self._hold_arm_targets()
        if self._draw_pen_target is None:
            self._fail_active_execution(
                'Arm writer entered PEN_SETTLE without a valid draw pen target.',
                params,
            )
            return
        if not self._pen_data_fresh(params.pen_contact_timeout_sec):
            self._fail_active_execution(
                'Pen contact/gap data are stale during PEN_SETTLE.',
                params,
            )
            return
        self._publish_pen(self._draw_pen_target)
        self._pen_settle_counter += 1
        if self._pen_settle_counter >= params.pen_settle_cycles:
            self._state = DRAW_SEGMENT

    def _handle_draw_segment(self, params: ArmWriterParams) -> None:
        primitive = self._current_primitive()
        segment = self._current_segment_points()
        if primitive is None or segment is None:
            self._fail_active_execution(
                'Arm writer lost the current segment during DRAW_SEGMENT.',
                params,
            )
            return

        if primitive.draw:
            self._publish_drawing_active(True)
            if self._draw_pen_target is None:
                self._fail_active_execution(
                    'Arm writer entered DRAW_SEGMENT without a draw pen target.',
                    params,
                )
                return
            self._publish_pen(self._draw_pen_target)
            if (
                params.contact_required_for_drawing
                and not self._effective_draw_contact(params)
            ):
                self._fail_active_execution(
                    'Drawing contact was lost or left the allowed gap range.',
                    params,
                )
                return
            speed = params.draw_speed
        else:
            self._publish_drawing_active(False)
            self._publish_pen(params.pen_up_pos)
            speed = params.travel_speed

        reached, error = self._move_pen_toward(segment[1], speed, params)
        if error is not None:
            self._fail_active_execution(error, params)
            return
        if not reached:
            return

        if self._segment_index + 1 < len(primitive.points) - 1:
            self._segment_index += 1
            self._state = ADVANCE_SEGMENT
            self._publish_drawing_active(False)
            return

        self._finish_current_primitive(params)

    def _handle_advance_segment(self, params: ArmWriterParams) -> None:
        primitive = self._current_primitive()
        if primitive is None:
            self._fail_active_execution(
                'Arm writer lost the current primitive during ADVANCE_SEGMENT.',
                params,
            )
            return

        self._publish_drawing_active(False)
        if primitive.draw and self._draw_pen_target is not None:
            self._publish_pen(self._draw_pen_target)
        else:
            self._publish_pen(params.pen_up_pos)
        self._hold_arm_targets()
        self._state = DRAW_SEGMENT

    def _handle_pen_up(self, params: ArmWriterParams) -> None:
        self._publish_drawing_active(False)
        self._publish_pen(params.pen_up_pos)
        self._hold_arm_targets()
        if self._pen_lift_started_sec is None:
            self._pen_lift_started_sec = self._now_sec()
            return
        if (
            self._now_sec() - self._pen_lift_started_sec
        ) < params.pen_lift_timeout_sec:
            return

        next_state = self._next_state_after_pen_up or DONE
        self._next_state_after_pen_up = None
        self._pen_lift_started_sec = None
        if next_state == DONE:
            self._state = DONE
            self._publish_drawing_active(False)
            self._publish_pen(params.pen_up_pos)
            self._set_status('done')
        else:
            self._state = next_state

    def _handle_done(self, params: ArmWriterParams) -> None:
        self._publish_drawing_active(False)
        self._publish_pen(params.pen_up_pos)
        self._set_status('done')

    def _handle_error(self, params: ArmWriterParams) -> None:
        self._publish_drawing_active(False)
        self._publish_pen(params.pen_up_pos)
        self._set_status('error')

    def _safe_idle(self, params: ArmWriterParams) -> None:
        self._publish_drawing_active(False)
        self._publish_pen(params.pen_up_pos)
        self._set_status('idle')
        self._state = IDLE

    def _on_timer(self) -> None:
        params = self._read_params()

        if self._stroke_executor_status == 'running' and self._state in ACTIVE_STATES:
            self._fail_active_execution(
                'Arm writer aborted because stroke_executor_status=running.',
                params,
            )
            return

        if not params.enabled:
            if self._enabled_last or self._state != IDLE:
                self._reset_execution()
                self._safe_idle(params)
            self._enabled_last = False
            return

        self._enabled_last = True
        self._maybe_start_pending_plan(params)

        if self._state in ACTIVE_STATES:
            if not self._pose_fresh(params.pose_timeout_sec):
                self._fail_active_execution(
                    'Robot pose is stale during arm writing.',
                    params,
                )
                return
            if self._body_drift_exceeded(params):
                self._fail_active_execution(
                    'Robot body drift exceeded Arm Writer V1 limits.',
                    params,
                )
                return

        if self._state == IDLE:
            self._publish_drawing_active(False)
            self._publish_pen(params.pen_up_pos)
            self._set_status('idle')
            return
        if self._state == MOVE_TO_START:
            self._handle_move_to_start(params)
            return
        if self._state == PEN_PROBE:
            self._handle_pen_probe(params)
            return
        if self._state == PEN_SETTLE:
            self._handle_pen_settle(params)
            return
        if self._state == DRAW_SEGMENT:
            self._handle_draw_segment(params)
            return
        if self._state == ADVANCE_SEGMENT:
            self._handle_advance_segment(params)
            return
        if self._state == PEN_UP:
            self._handle_pen_up(params)
            return
        if self._state == DONE:
            self._handle_done(params)
            return
        if self._state == ERROR:
            self._handle_error(params)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmWriterController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
