"""Board-frame arm point controller and local arm-only writer.

This node supports two mutually exclusive execution modes while enabled:
- a single board-frame point target on /wall_climber/arm_pose_target
- a small local stroke plan on /wall_climber/arm_stroke_plan

Both modes publish shoulder targets on /wall_climber/arm_joint_targets and
use the existing pen target path without commanding any body motion.
"""

from dataclasses import dataclass
import json
import math

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String


TIMER_PERIOD_SEC = 0.05  # 20 Hz
_AXIS_EPS = 1.0e-6


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class PoseSnapshot:
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class ControllerParams:
    enabled: bool
    pose_timeout_sec: float
    pen_pose_timeout_sec: float
    pen_up_pos: float
    pen_down_pos: float
    target_reached_tol: float
    ik_verify_tol: float
    local_x_min: float
    local_x_max: float
    local_y_min: float
    local_y_max: float
    draw_sample_step_m: float
    pen_settle_sec: float
    pen_contact_timeout_sec: float


@dataclass(frozen=True)
class ActiveGoal:
    board_x: float
    board_y: float
    body_x: float
    body_y: float
    local_x: float
    local_y: float
    theta_l: float
    theta_r: float


@dataclass(frozen=True)
class StrokeSegment:
    goals: tuple[ActiveGoal, ...]


class ArmPoseController(Node):
    _STATUS_IDLE = 'idle'
    _STATUS_MOVING_TO_START = 'moving_to_start'
    _STATUS_DRAWING = 'drawing'
    _STATUS_DONE = 'done'
    _STATUS_ERROR = 'error'

    _MODE_POINT = 'point'
    _MODE_STROKE = 'stroke'

    _STROKE_MOVE_TO_START = 'move_to_start'
    _STROKE_PEN_DOWN = 'pen_down'
    _STROKE_DRAW_SEGMENT = 'draw_segment'
    _STROKE_PEN_UP = 'pen_up'

    _LEFT_SHOULDER_NAME = 'left_shoulder_joint'
    _RIGHT_SHOULDER_NAME = 'right_shoulder_joint'

    # Must match the current pantograph arm model used by the keyboard bridge.
    _SH_HALF = 0.05
    _UPPER_LEN = 0.14
    _FORE_LEN = 0.18
    _FORE_ANG = 0.281
    _ARM_ANCHOR_Y = 0.10
    _SHOULDER_MIN = -2.30
    _SHOULDER_MAX = 2.30
    _ELBOW_MIN = -2.349
    _ELBOW_MAX = 2.349

    def __init__(self) -> None:
        super().__init__('arm_pose_controller')

        self.declare_parameter('enabled', False)
        self.declare_parameter('pose_timeout_sec', 0.5)
        self.declare_parameter('pen_pose_timeout_sec', 0.5)
        self.declare_parameter('pen_up_pos', 0.018)
        self.declare_parameter('pen_down_pos', -0.020)
        self.declare_parameter('target_reached_tol', 0.010)
        self.declare_parameter('ik_verify_tol', 0.002)
        self.declare_parameter('local_x_min', -0.09)
        self.declare_parameter('local_x_max', 0.09)
        self.declare_parameter('local_y_min', 0.22)
        self.declare_parameter('local_y_max', 0.32)
        self.declare_parameter('draw_sample_step_m', 0.005)
        self.declare_parameter('pen_settle_sec', 0.15)
        self.declare_parameter('pen_contact_timeout_sec', 0.5)

        self._pose: Pose2D | None = None
        self._pose_stamp = None
        self._pen_x: float | None = None
        self._pen_y: float | None = None
        self._pen_pose_stamp = None
        self._pen_contact = False
        self._pen_contact_stamp = None

        self._status: str | None = None
        self._enabled_last = False
        self._mode: str | None = None
        self._active_goal: ActiveGoal | None = None
        self._current_command_pair: tuple[float, float] | None = None
        self._last_accepted_command_pair: tuple[float, float] | None = None

        self._stroke_segments: list[StrokeSegment] = []
        self._stroke_count = 0
        self._stroke_segment_index = 0
        self._stroke_waypoint_index = 0
        self._stroke_state: str | None = None
        self._stroke_state_enter_sec: float | None = None
        self._stroke_pose_snapshot: PoseSnapshot | None = None

        self.create_subscription(
            PointStamped,
            '/wall_climber/arm_pose_target',
            self._target_cb,
            10,
        )
        self.create_subscription(
            String,
            '/wall_climber/arm_stroke_plan',
            self._stroke_plan_cb,
            10,
        )
        self.create_subscription(
            Pose2D,
            '/wall_climber/robot_pose_board',
            self._pose_cb,
            10,
        )
        self.create_subscription(
            PointStamped,
            '/wall_climber/pen_pose_board',
            self._pen_pose_cb,
            10,
        )
        self.create_subscription(
            Bool,
            '/wall_climber/pen_contact',
            self._pen_contact_cb,
            10,
        )

        self._arm_target_pub = self.create_publisher(
            JointState, '/wall_climber/arm_joint_targets', 10
        )
        self._pen_pub = self.create_publisher(
            Float64, '/wall_climber/pen_target', 10
        )
        self._status_pub = self.create_publisher(
            String, '/wall_climber/arm_pose_status', 10
        )

        self._timer = self.create_timer(TIMER_PERIOD_SEC, self._on_timer)
        self._set_status(self._STATUS_IDLE)
        self.get_logger().info('Arm pose controller ready (enabled=false).')

    def _read_params(self) -> ControllerParams:
        return ControllerParams(
            enabled=bool(self.get_parameter('enabled').value),
            pose_timeout_sec=float(self.get_parameter('pose_timeout_sec').value),
            pen_pose_timeout_sec=float(
                self.get_parameter('pen_pose_timeout_sec').value
            ),
            pen_up_pos=float(self.get_parameter('pen_up_pos').value),
            pen_down_pos=float(self.get_parameter('pen_down_pos').value),
            target_reached_tol=float(
                self.get_parameter('target_reached_tol').value
            ),
            ik_verify_tol=float(self.get_parameter('ik_verify_tol').value),
            local_x_min=float(self.get_parameter('local_x_min').value),
            local_x_max=float(self.get_parameter('local_x_max').value),
            local_y_min=float(self.get_parameter('local_y_min').value),
            local_y_max=float(self.get_parameter('local_y_max').value),
            draw_sample_step_m=float(self.get_parameter('draw_sample_step_m').value),
            pen_settle_sec=float(self.get_parameter('pen_settle_sec').value),
            pen_contact_timeout_sec=float(
                self.get_parameter('pen_contact_timeout_sec').value
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

    def _target_cb(self, msg: PointStamped) -> None:
        params = self._read_params()
        if not params.enabled:
            return
        if self._execution_active():
            self.get_logger().warn(
                'Arm pose target rejected: controller is busy with an active arm command.'
            )
            return

        frame_id = str(msg.header.frame_id).strip()
        if frame_id != 'board':
            self._enter_error(
                'Arm pose target rejected: frame_id must be "board".',
                params,
            )
            return

        try:
            board_x = float(msg.point.x)
            board_y = float(msg.point.y)
        except (TypeError, ValueError):
            self._enter_error(
                'Arm pose target rejected: board target must be finite.',
                params,
            )
            return
        if not (math.isfinite(board_x) and math.isfinite(board_y)):
            self._enter_error(
                'Arm pose target rejected: board target must be finite.',
                params,
            )
            return

        if not self._pose_fresh(params.pose_timeout_sec):
            self._enter_error(
                'Arm pose target rejected: robot pose is not fresh.',
                params,
            )
            return

        pose_snapshot = self._current_pose_snapshot()
        if pose_snapshot is None:
            self._enter_error(
                'Arm pose target rejected: robot pose is unavailable.',
                params,
            )
            return

        goal, reason = self._goal_from_pose_snapshot(
            board_x,
            board_y,
            params,
            pose_snapshot,
        )
        if goal is None:
            self._enter_error(reason, params)
            return

        self._begin_point_goal(goal, params)
        self.get_logger().info(
            'Accepted arm pose target '
            f'board=({goal.board_x:.4f}, {goal.board_y:.4f}) '
            f'body_local=({goal.body_x:.4f}, {goal.body_y:.4f}) '
            f'local=({goal.local_x:.4f}, {goal.local_y:.4f}) '
            f'shoulders=({goal.theta_l:.4f}, {goal.theta_r:.4f})'
        )

    def _stroke_plan_cb(self, msg: String) -> None:
        params = self._read_params()
        if not params.enabled:
            return
        if self._execution_active():
            self.get_logger().warn(
                'Arm stroke plan rejected: controller is busy with an active arm command.'
            )
            return
        if not self._pose_fresh(params.pose_timeout_sec):
            self._enter_error(
                'Arm stroke plan rejected: robot pose is not fresh.',
                params,
            )
            return
        if not self._pen_pose_fresh(params.pen_pose_timeout_sec):
            self._enter_error(
                'Arm stroke plan rejected: pen pose is not fresh.',
                params,
            )
            return

        pose_snapshot = self._current_pose_snapshot()
        if pose_snapshot is None:
            self._enter_error(
                'Arm stroke plan rejected: robot pose is unavailable.',
                params,
            )
            return

        parsed = self._parse_stroke_plan(msg.data, params, pose_snapshot)
        if parsed is None:
            return

        stroke_count, segments = parsed
        self._begin_stroke_plan(stroke_count, segments, pose_snapshot, params)
        self.get_logger().info(
            f'Arm stroke plan accepted: strokes={stroke_count} segments={len(segments)}'
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

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

    def _publish_shoulder_targets(self, theta_l: float, theta_r: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self._LEFT_SHOULDER_NAME, self._RIGHT_SHOULDER_NAME]
        msg.position = [float(theta_l), float(theta_r)]
        self._arm_target_pub.publish(msg)

    def _pose_fresh(self, timeout_sec: float) -> bool:
        if self._pose is None or self._pose_stamp is None:
            return False
        age_sec = (self.get_clock().now() - self._pose_stamp).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_pose_fresh(self, timeout_sec: float) -> bool:
        if self._pen_x is None or self._pen_y is None or self._pen_pose_stamp is None:
            return False
        age_sec = (
            self.get_clock().now() - self._pen_pose_stamp
        ).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _pen_contact_fresh(self, timeout_sec: float) -> bool:
        if self._pen_contact_stamp is None:
            return False
        age_sec = (
            self.get_clock().now() - self._pen_contact_stamp
        ).nanoseconds * 1e-9
        return age_sec <= timeout_sec

    def _current_pose_snapshot(self) -> PoseSnapshot | None:
        if self._pose is None:
            return None
        return PoseSnapshot(
            x=float(self._pose.x),
            y=float(self._pose.y),
            theta=float(self._pose.theta),
        )

    def _hold_command_pair(self) -> tuple[float, float] | None:
        if self._current_command_pair is not None:
            return self._current_command_pair
        return self._last_accepted_command_pair

    def _publish_hold_if_available(self) -> None:
        hold_pair = self._hold_command_pair()
        if hold_pair is None:
            return
        self._publish_shoulder_targets(hold_pair[0], hold_pair[1])

    def _execution_active(self) -> bool:
        return self._mode is not None

    def _clear_execution(self) -> None:
        self._mode = None
        self._active_goal = None
        self._stroke_segments = []
        self._stroke_count = 0
        self._stroke_segment_index = 0
        self._stroke_waypoint_index = 0
        self._stroke_state = None
        self._stroke_state_enter_sec = None
        self._stroke_pose_snapshot = None

    def _enter_error(self, reason: str, params: ControllerParams) -> None:
        self._clear_execution()
        self._publish_pen(params.pen_up_pos)
        self._set_status(self._STATUS_ERROR)
        self.get_logger().warn(reason)

    def _activate_goal(self, goal: ActiveGoal) -> None:
        self._active_goal = goal
        self._current_command_pair = (goal.theta_l, goal.theta_r)
        self._last_accepted_command_pair = self._current_command_pair

    def _begin_point_goal(self, goal: ActiveGoal, params: ControllerParams) -> None:
        self._clear_execution()
        self._mode = self._MODE_POINT
        self._activate_goal(goal)
        self._publish_pen(params.pen_up_pos)
        self._publish_shoulder_targets(goal.theta_l, goal.theta_r)
        self._set_status(self._STATUS_MOVING_TO_START)

    def _begin_stroke_plan(
        self,
        stroke_count: int,
        segments: list[StrokeSegment],
        pose_snapshot: PoseSnapshot,
        params: ControllerParams,
    ) -> None:
        self._clear_execution()
        self._mode = self._MODE_STROKE
        self._stroke_count = stroke_count
        self._stroke_segments = segments
        self._stroke_segment_index = 0
        self._stroke_waypoint_index = 0
        self._stroke_pose_snapshot = pose_snapshot
        self._activate_goal(segments[0].goals[0])
        self._set_stroke_state(self._STROKE_MOVE_TO_START, params)
        self._publish_pen(params.pen_up_pos)
        self._publish_shoulder_targets(
            segments[0].goals[0].theta_l,
            segments[0].goals[0].theta_r,
        )

    def _set_stroke_state(self, state: str, params: ControllerParams) -> None:
        self._stroke_state = state
        self._stroke_state_enter_sec = self._now_sec()
        if state == self._STROKE_MOVE_TO_START:
            self._set_status(self._STATUS_MOVING_TO_START)
            self.get_logger().info(
                f'Entered move_to_start for arm stroke segment {self._stroke_segment_index + 1}/{len(self._stroke_segments)}'
            )
        elif state == self._STROKE_DRAW_SEGMENT:
            self._set_status(self._STATUS_DRAWING)
            self.get_logger().info(
                f'Entered drawing for arm stroke segment {self._stroke_segment_index + 1}/{len(self._stroke_segments)}'
            )

    def _distance_to_board_point(self, board_x: float, board_y: float) -> float:
        assert self._pen_x is not None and self._pen_y is not None
        return math.hypot(self._pen_x - board_x, self._pen_y - board_y)

    def _board_to_arm_local_from_pose(
        self,
        pose_snapshot: PoseSnapshot,
        board_x: float,
        board_y: float,
    ) -> tuple[float, float, float, float]:
        dx = board_x - pose_snapshot.x
        # board +y is down, while body/arm local +y is up toward the arm workspace.
        # Only the board-y sign is inverted here; the rotation stays consistent with
        # robot_pose_board.theta = atan2(board_down, board_right).
        dy_up = pose_snapshot.y - board_y
        cos_theta = math.cos(pose_snapshot.theta)
        sin_theta = math.sin(pose_snapshot.theta)
        body_x = cos_theta * dx - sin_theta * dy_up
        body_y = sin_theta * dx + cos_theta * dy_up
        return body_x, body_y, body_x, body_y - self._ARM_ANCHOR_Y

    def _inside_local_workspace(
        self, local_x: float, local_y: float, params: ControllerParams
    ) -> bool:
        return (
            params.local_x_min <= local_x <= params.local_x_max
            and params.local_y_min <= local_y <= params.local_y_max
        )

    def _inside_two_link_bounds(self, local_x: float, local_y: float) -> bool:
        lower = abs(self._UPPER_LEN - self._FORE_LEN) - 1.0e-6
        upper = self._UPPER_LEN + self._FORE_LEN + 1.0e-6
        for shoulder_x in (-self._SH_HALF, self._SH_HALF):
            distance = math.hypot(local_x - shoulder_x, local_y)
            if distance < lower or distance > upper:
                return False
        return True

    def _reference_command_pair(self) -> tuple[float, float]:
        hold_pair = self._hold_command_pair()
        if hold_pair is not None:
            return hold_pair
        return (0.0, 0.0)

    def _shoulder_candidates(
        self, target_x: float, target_y: float, shoulder_x: float
    ) -> list[float]:
        dx = target_x - shoulder_x
        dy = target_y
        distance = math.hypot(dx, dy)
        if distance < 1.0e-9:
            return []

        cos_delta = (
            self._UPPER_LEN * self._UPPER_LEN
            + distance * distance
            - self._FORE_LEN * self._FORE_LEN
        ) / (2.0 * self._UPPER_LEN * distance)
        if cos_delta < -1.0 - 1.0e-9 or cos_delta > 1.0 + 1.0e-9:
            return []

        cos_delta = _clamp(cos_delta, -1.0, 1.0)
        gamma = math.atan2(dy, dx)
        delta = math.acos(cos_delta)
        candidates: list[float] = []
        for sign in (1.0, -1.0):
            alpha = gamma + sign * delta
            theta = _wrap_to_pi(alpha - math.pi / 2.0)
            if self._SHOULDER_MIN <= theta <= self._SHOULDER_MAX:
                if not any(abs(theta - existing) < 1.0e-9 for existing in candidates):
                    candidates.append(theta)
        return candidates

    def _solve_closed_loop(
        self, theta_l: float, theta_r: float
    ) -> tuple[float, float, float, float] | None:
        s_lx, s_ly = -self._SH_HALF, 0.0
        s_rx, s_ry = self._SH_HALF, 0.0

        e_lx = s_lx + self._UPPER_LEN * (-math.sin(theta_l))
        e_ly = s_ly + self._UPPER_LEN * math.cos(theta_l)
        e_rx = s_rx + self._UPPER_LEN * (-math.sin(theta_r))
        e_ry = s_ry + self._UPPER_LEN * math.cos(theta_r)

        dx = e_rx - e_lx
        dy = e_ry - e_ly
        distance = math.hypot(dx, dy)
        if distance < 1.0e-9 or distance > 2.0 * self._FORE_LEN:
            return None

        a = distance / 2.0
        h_sq = self._FORE_LEN * self._FORE_LEN - a * a
        if h_sq < 0.0:
            return None

        h = math.sqrt(h_sq)
        mx = (e_lx + e_rx) / 2.0
        my = (e_ly + e_ry) / 2.0
        ux = dx / distance
        uy = dy / distance
        nx = -uy
        ny = ux

        p1x = mx + h * nx
        p1y = my + h * ny
        p2x = mx - h * nx
        p2y = my - h * ny
        if p1y >= p2y:
            px, py = p1x, p1y
        else:
            px, py = p2x, p2y

        alpha_l = math.pi / 2.0 + theta_l
        alpha_r = math.pi / 2.0 + theta_r
        beta_l = math.atan2(py - e_ly, px - e_lx)
        beta_r = math.atan2(py - e_ry, px - e_rx)

        phi_l = _wrap_to_pi(beta_l - alpha_l + self._FORE_ANG)
        phi_r = _wrap_to_pi(beta_r - alpha_r - self._FORE_ANG)
        return px, py, phi_l, phi_r

    def _verify_candidate(
        self,
        theta_l: float,
        theta_r: float,
        target_x: float,
        target_y: float,
        tol: float,
    ) -> bool:
        solved = self._solve_closed_loop(theta_l, theta_r)
        if solved is None:
            return False
        px, py, phi_l, phi_r = solved
        if not (self._ELBOW_MIN <= phi_l <= self._ELBOW_MAX):
            return False
        if not (self._ELBOW_MIN <= phi_r <= self._ELBOW_MAX):
            return False
        return math.hypot(px - target_x, py - target_y) <= tol

    def _best_shoulder_pair(
        self,
        target_x: float,
        target_y: float,
        verify_tol: float,
        reference_pair: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        left_candidates = self._shoulder_candidates(
            target_x, target_y, -self._SH_HALF
        )
        right_candidates = self._shoulder_candidates(
            target_x, target_y, self._SH_HALF
        )
        if not left_candidates or not right_candidates:
            return None

        verified_pairs: list[tuple[float, float]] = []
        for theta_l in left_candidates:
            for theta_r in right_candidates:
                if self._verify_candidate(theta_l, theta_r, target_x, target_y, verify_tol):
                    verified_pairs.append((theta_l, theta_r))

        if not verified_pairs:
            return None

        ref_l, ref_r = reference_pair or self._reference_command_pair()

        def score(pair: tuple[float, float]) -> tuple[float, float]:
            d_l = _wrap_to_pi(pair[0] - ref_l)
            d_r = _wrap_to_pi(pair[1] - ref_r)
            continuity_cost = d_l * d_l + d_r * d_r
            magnitude_cost = pair[0] * pair[0] + pair[1] * pair[1]
            return continuity_cost, magnitude_cost

        verified_pairs.sort(key=score)
        return verified_pairs[0]

    def _goal_from_pose_snapshot(
        self,
        board_x: float,
        board_y: float,
        params: ControllerParams,
        pose_snapshot: PoseSnapshot,
        reference_pair: tuple[float, float] | None = None,
    ) -> tuple[ActiveGoal | None, str]:
        body_x, body_y, local_x, local_y = self._board_to_arm_local_from_pose(
            pose_snapshot,
            board_x,
            board_y,
        )
        if not self._inside_local_workspace(local_x, local_y, params):
            return None, (
                'Arm target rejected: outside conservative local workspace. '
                f'board=({board_x:.4f}, {board_y:.4f}) '
                f'local=({local_x:.4f}, {local_y:.4f})'
            )
        if not self._inside_two_link_bounds(local_x, local_y):
            return None, (
                'Arm target rejected: outside arm reach bounds. '
                f'board=({board_x:.4f}, {board_y:.4f}) '
                f'local=({local_x:.4f}, {local_y:.4f})'
            )

        pair = self._best_shoulder_pair(
            local_x,
            local_y,
            params.ik_verify_tol,
            reference_pair=reference_pair,
        )
        if pair is None:
            return None, 'Arm target rejected: no verified shoulder solution exists.'

        return ActiveGoal(
            board_x=board_x,
            board_y=board_y,
            body_x=body_x,
            body_y=body_y,
            local_x=local_x,
            local_y=local_y,
            theta_l=pair[0],
            theta_r=pair[1],
        ), ''

    def _parse_points(self, stroke: object, stroke_index: int) -> tuple[list[tuple[float, float]] | None, str | None]:
        if not isinstance(stroke, dict):
            return None, f'stroke {stroke_index} is not an object'
        points = stroke.get('points')
        if not isinstance(points, list):
            return None, f'stroke {stroke_index} points must be a list'
        parsed: list[tuple[float, float]] = []
        for point_index, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None, f'stroke {stroke_index} point {point_index} must be [x, y]'
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                return (
                    None,
                    f'stroke {stroke_index} point {point_index} must be finite',
                )
            if not math.isfinite(x) or not math.isfinite(y):
                return None, f'stroke {stroke_index} point {point_index} must be finite'
            parsed.append((x, y))
        return parsed, None

    def _sample_segment_goals(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        params: ControllerParams,
        pose_snapshot: PoseSnapshot,
        reference_pair: tuple[float, float],
        stroke_index: int,
        segment_index: int,
    ) -> tuple[list[ActiveGoal] | None, tuple[float, float], str | None]:
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        if abs(dx) <= _AXIS_EPS and abs(dy) <= _AXIS_EPS:
            return None, reference_pair, (
                f'stroke {stroke_index} segment {segment_index} is zero-length'
            )
        if abs(dx) > _AXIS_EPS and abs(dy) > _AXIS_EPS:
            return None, reference_pair, (
                f'stroke {stroke_index} segment {segment_index} is diagonal and unsupported'
            )

        segment_length = abs(dx) + abs(dy)
        sample_step = max(params.draw_sample_step_m, 1.0e-4)
        steps = max(1, int(math.ceil(segment_length / sample_step)))
        sampled_goals: list[ActiveGoal] = []
        local_reference = reference_pair
        for step_index in range(steps + 1):
            t = step_index / steps
            board_x = start_point[0] + dx * t
            board_y = start_point[1] + dy * t
            goal, reason = self._goal_from_pose_snapshot(
                board_x,
                board_y,
                params,
                pose_snapshot,
                reference_pair=local_reference,
            )
            if goal is None:
                return None, local_reference, (
                    f'stroke {stroke_index} segment {segment_index} invalid: {reason}'
                )
            sampled_goals.append(goal)
            local_reference = (goal.theta_l, goal.theta_r)
        return sampled_goals, local_reference, None

    def _parse_stroke_plan(
        self,
        payload: str,
        params: ControllerParams,
        pose_snapshot: PoseSnapshot,
    ) -> tuple[int, list[StrokeSegment]] | None:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            self._enter_error(f'Arm stroke plan rejected: invalid JSON ({exc.msg}).', params)
            return None

        if not isinstance(data, dict):
            self._enter_error('Arm stroke plan rejected: top-level JSON must be an object.', params)
            return None
        if data.get('frame') != 'board':
            self._enter_error('Arm stroke plan rejected: frame must be "board".', params)
            return None

        strokes = data.get('strokes')
        if not isinstance(strokes, list) or not strokes:
            self._enter_error('Arm stroke plan rejected: strokes must be a non-empty list.', params)
            return None

        segments: list[StrokeSegment] = []
        reference_pair = self._reference_command_pair()
        for stroke_index, stroke in enumerate(strokes):
            if not isinstance(stroke, dict):
                self._enter_error(
                    f'Arm stroke plan rejected: stroke {stroke_index} must be an object.',
                    params,
                )
                return None

            stroke_type = stroke.get('type')
            if stroke_type not in ('line', 'polyline'):
                self._enter_error(
                    f'Arm stroke plan rejected: stroke {stroke_index} has unsupported type.',
                    params,
                )
                return None
            if stroke.get('draw') is not True:
                self._enter_error(
                    f'Arm stroke plan rejected: stroke {stroke_index} must set draw=true.',
                    params,
                )
                return None

            points, reason = self._parse_points(stroke, stroke_index)
            if points is None:
                self._enter_error(f'Arm stroke plan rejected: {reason}.', params)
                return None

            if stroke_type == 'line' and len(points) != 2:
                self._enter_error(
                    f'Arm stroke plan rejected: stroke {stroke_index} line must contain exactly 2 points.',
                    params,
                )
                return None
            if stroke_type == 'polyline' and len(points) < 2:
                self._enter_error(
                    f'Arm stroke plan rejected: stroke {stroke_index} polyline must contain at least 2 points.',
                    params,
                )
                return None

            for segment_index in range(len(points) - 1):
                segment_goals, reference_pair, reason = self._sample_segment_goals(
                    points[segment_index],
                    points[segment_index + 1],
                    params,
                    pose_snapshot,
                    reference_pair,
                    stroke_index,
                    segment_index,
                )
                if segment_goals is None:
                    self._enter_error(f'Arm stroke plan rejected: {reason}.', params)
                    return None
                segments.append(StrokeSegment(goals=tuple(segment_goals)))

        if not segments:
            self._enter_error('Arm stroke plan rejected: no executable segments found.', params)
            return None
        return len(strokes), segments

    def _current_segment(self) -> StrokeSegment | None:
        if not self._stroke_segments:
            return None
        if self._stroke_segment_index < 0 or self._stroke_segment_index >= len(self._stroke_segments):
            return None
        return self._stroke_segments[self._stroke_segment_index]

    def _finish_point_goal(self, params: ControllerParams) -> None:
        self._clear_execution()
        self._publish_pen(params.pen_up_pos)
        self._set_status(self._STATUS_DONE)
        self.get_logger().info('Arm pose target done.')

    def _finish_stroke_plan(self, params: ControllerParams) -> None:
        self._clear_execution()
        self._publish_pen(params.pen_up_pos)
        self._set_status(self._STATUS_DONE)
        self.get_logger().info('Arm stroke plan done.')

    def _run_point_mode(self, params: ControllerParams) -> None:
        if self._active_goal is None:
            self._enter_error('Arm pose controller error: missing active point goal.', params)
            return
        if not self._pose_fresh(params.pose_timeout_sec):
            self._enter_error(
                'Arm pose controller error: robot pose went stale during point motion.',
                params,
            )
            return
        if not self._pen_pose_fresh(params.pen_pose_timeout_sec):
            self._enter_error(
                'Arm pose controller error: pen pose went stale during point motion.',
                params,
            )
            return

        self._publish_pen(params.pen_up_pos)
        self._publish_shoulder_targets(
            self._active_goal.theta_l,
            self._active_goal.theta_r,
        )

        if (
            self._distance_to_board_point(
                self._active_goal.board_x,
                self._active_goal.board_y,
            )
            <= params.target_reached_tol
        ):
            self._finish_point_goal(params)
        else:
            self._set_status(self._STATUS_MOVING_TO_START)

    def _run_stroke_mode(self, params: ControllerParams) -> None:
        if self._active_goal is None:
            self._enter_error('Arm stroke execution error: missing active goal.', params)
            return
        if not self._pose_fresh(params.pose_timeout_sec):
            self._enter_error(
                'Arm stroke execution error: robot pose went stale during stroke execution.',
                params,
            )
            return
        if not self._pen_pose_fresh(params.pen_pose_timeout_sec):
            self._enter_error(
                'Arm stroke execution error: pen pose went stale during stroke execution.',
                params,
            )
            return

        segment = self._current_segment()
        if segment is None:
            self._enter_error('Arm stroke execution error: invalid segment state.', params)
            return
        if self._stroke_state_enter_sec is None or self._stroke_state is None:
            self._enter_error('Arm stroke execution error: invalid stroke state.', params)
            return

        now_sec = self._now_sec()

        if self._stroke_state == self._STROKE_MOVE_TO_START:
            self._publish_pen(params.pen_up_pos)
            self._publish_shoulder_targets(
                self._active_goal.theta_l,
                self._active_goal.theta_r,
            )
            if (
                self._distance_to_board_point(
                    self._active_goal.board_x,
                    self._active_goal.board_y,
                )
                <= params.target_reached_tol
            ):
                self._set_stroke_state(self._STROKE_PEN_DOWN, params)
            return

        if self._stroke_state == self._STROKE_PEN_DOWN:
            self._publish_pen(params.pen_down_pos)
            self._publish_shoulder_targets(
                self._active_goal.theta_l,
                self._active_goal.theta_r,
            )
            elapsed = now_sec - self._stroke_state_enter_sec
            if elapsed < params.pen_settle_sec:
                return
            if (
                self._pen_contact_fresh(params.pen_contact_timeout_sec)
                and self._pen_contact
            ):
                if len(segment.goals) <= 1:
                    self._set_stroke_state(self._STROKE_PEN_UP, params)
                    return
                self._stroke_waypoint_index = 1
                self._activate_goal(segment.goals[self._stroke_waypoint_index])
                self._set_stroke_state(self._STROKE_DRAW_SEGMENT, params)
                return
            if elapsed >= (params.pen_settle_sec + params.pen_contact_timeout_sec):
                self._enter_error(
                    'Arm stroke execution error: pen contact was not confirmed before drawing.',
                    params,
                )
            return

        if self._stroke_state == self._STROKE_DRAW_SEGMENT:
            self._publish_pen(params.pen_down_pos)
            self._publish_shoulder_targets(
                self._active_goal.theta_l,
                self._active_goal.theta_r,
            )
            if (
                self._distance_to_board_point(
                    self._active_goal.board_x,
                    self._active_goal.board_y,
                )
                > params.target_reached_tol
            ):
                self._set_status(self._STATUS_DRAWING)
                return

            if self._stroke_waypoint_index + 1 < len(segment.goals):
                self._stroke_waypoint_index += 1
                self._activate_goal(segment.goals[self._stroke_waypoint_index])
                return

            self._set_stroke_state(self._STROKE_PEN_UP, params)
            return

        if self._stroke_state == self._STROKE_PEN_UP:
            self._publish_pen(params.pen_up_pos)
            self._publish_shoulder_targets(
                self._active_goal.theta_l,
                self._active_goal.theta_r,
            )
            if (now_sec - self._stroke_state_enter_sec) < params.pen_settle_sec:
                return
            if self._stroke_segment_index + 1 < len(self._stroke_segments):
                self._stroke_segment_index += 1
                self._stroke_waypoint_index = 0
                next_segment = self._current_segment()
                if next_segment is None:
                    self._enter_error('Arm stroke execution error: invalid next segment state.', params)
                    return
                self._activate_goal(next_segment.goals[0])
                self._set_stroke_state(self._STROKE_MOVE_TO_START, params)
                return
            self._finish_stroke_plan(params)
            return

        self._enter_error('Arm stroke execution error: unknown stroke state.', params)

    def _on_timer(self) -> None:
        params = self._read_params()

        if not params.enabled:
            if self._enabled_last:
                self._clear_execution()
                self._publish_pen(params.pen_up_pos)
                self._set_status(self._STATUS_IDLE)
            self._enabled_last = False
            return

        self._enabled_last = True

        if self._mode == self._MODE_POINT:
            self._run_point_mode(params)
            return
        if self._mode == self._MODE_STROKE:
            self._run_stroke_mode(params)
            return

        self._publish_pen(params.pen_up_pos)
        self._publish_hold_if_available()
        if self._status is None:
            self._set_status(self._STATUS_IDLE)
        elif self._status not in (self._STATUS_DONE, self._STATUS_ERROR):
            self._set_status(self._STATUS_IDLE)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmPoseController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
