"""Point-to-point board-frame arm pose controller.

This node accepts a single board-frame target point, converts it into
the pantograph arm's local workspace, and publishes shoulder targets on
/wall_climber/arm_joint_targets while keeping the pen safely raised.
"""

from dataclasses import dataclass
import math

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String


TIMER_PERIOD_SEC = 0.05  # 20 Hz


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class ControllerParams:
    enabled: bool
    pose_timeout_sec: float
    pen_pose_timeout_sec: float
    pen_up_pos: float
    target_reached_tol: float
    ik_verify_tol: float
    local_x_min: float
    local_x_max: float
    local_y_min: float
    local_y_max: float


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


class ArmPoseController(Node):
    _STATUS_IDLE = 'idle'
    _STATUS_MOVING = 'moving'
    _STATUS_REACHED = 'reached'
    _STATUS_ERROR = 'error'

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
        self.declare_parameter('target_reached_tol', 0.010)
        self.declare_parameter('ik_verify_tol', 0.002)
        self.declare_parameter('local_x_min', -0.09)
        self.declare_parameter('local_x_max', 0.09)
        self.declare_parameter('local_y_min', 0.22)
        self.declare_parameter('local_y_max', 0.32)

        self._pose: Pose2D | None = None
        self._pose_stamp = None
        self._pen_x: float | None = None
        self._pen_y: float | None = None
        self._pen_pose_stamp = None

        self._status: str | None = None
        self._enabled_last = False
        self._active_goal: ActiveGoal | None = None
        self._current_command_pair: tuple[float, float] | None = None
        self._last_accepted_command_pair: tuple[float, float] | None = None

        self.create_subscription(
            PointStamped,
            '/wall_climber/arm_pose_target',
            self._target_cb,
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
            target_reached_tol=float(
                self.get_parameter('target_reached_tol').value
            ),
            ik_verify_tol=float(self.get_parameter('ik_verify_tol').value),
            local_x_min=float(self.get_parameter('local_x_min').value),
            local_x_max=float(self.get_parameter('local_x_max').value),
            local_y_min=float(self.get_parameter('local_y_min').value),
            local_y_max=float(self.get_parameter('local_y_max').value),
        )

    def _pose_cb(self, msg: Pose2D) -> None:
        self._pose = msg
        self._pose_stamp = self.get_clock().now()

    def _pen_pose_cb(self, msg: PointStamped) -> None:
        self._pen_x = float(msg.point.x)
        self._pen_y = float(msg.point.y)
        self._pen_pose_stamp = self.get_clock().now()

    def _target_cb(self, msg: PointStamped) -> None:
        params = self._read_params()
        if not params.enabled:
            return

        frame_id = str(msg.header.frame_id).strip()
        if frame_id != 'board':
            self._enter_error(
                'Arm pose target rejected: frame_id must be "board".',
                params,
            )
            return

        board_x = float(msg.point.x)
        board_y = float(msg.point.y)
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

        goal = self._build_goal(board_x, board_y, params)
        if goal is None:
            return

        self._active_goal = goal
        self._current_command_pair = (goal.theta_l, goal.theta_r)
        self._last_accepted_command_pair = self._current_command_pair
        self._publish_pen(params.pen_up_pos)
        self._publish_shoulder_targets(goal.theta_l, goal.theta_r)
        self._set_status(self._STATUS_MOVING)
        self.get_logger().info(
            'Accepted arm pose target '
            f'board=({goal.board_x:.4f}, {goal.board_y:.4f}) '
            f'body_local=({goal.body_x:.4f}, {goal.body_y:.4f}) '
            f'local=({goal.local_x:.4f}, {goal.local_y:.4f}) '
            f'shoulders=({goal.theta_l:.4f}, {goal.theta_r:.4f})'
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

    def _clear_active_goal(self) -> None:
        self._active_goal = None
        self._current_command_pair = None

    def _enter_error(self, reason: str, params: ControllerParams) -> None:
        self._clear_active_goal()
        self._publish_pen(params.pen_up_pos)
        self._set_status(self._STATUS_ERROR)
        self.get_logger().warn(reason)

    def _board_to_arm_local(
        self, board_x: float, board_y: float
    ) -> tuple[float, float, float, float]:
        assert self._pose is not None
        dx = board_x - float(self._pose.x)
        # board +y is down, while body/arm local +y is up toward the arm workspace.
        # Only the board-y sign is inverted here; the rotation stays consistent with
        # robot_pose_board.theta = atan2(board_down, board_right).
        dy_up = float(self._pose.y) - board_y
        theta = float(self._pose.theta)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
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
        if self._current_command_pair is not None:
            return self._current_command_pair
        if self._last_accepted_command_pair is not None:
            return self._last_accepted_command_pair
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
        self, theta_l: float, theta_r: float, target_x: float, target_y: float, tol: float
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
        self, target_x: float, target_y: float, verify_tol: float
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

        ref_l, ref_r = self._reference_command_pair()

        def score(pair: tuple[float, float]) -> tuple[float, float]:
            d_l = _wrap_to_pi(pair[0] - ref_l)
            d_r = _wrap_to_pi(pair[1] - ref_r)
            continuity_cost = d_l * d_l + d_r * d_r
            magnitude_cost = pair[0] * pair[0] + pair[1] * pair[1]
            return continuity_cost, magnitude_cost

        verified_pairs.sort(key=score)
        return verified_pairs[0]

    def _build_goal(
        self, board_x: float, board_y: float, params: ControllerParams
    ) -> ActiveGoal | None:
        body_x, body_y, local_x, local_y = self._board_to_arm_local(board_x, board_y)
        if not self._inside_local_workspace(local_x, local_y, params):
            self._enter_error(
                'Arm pose target rejected: outside conservative local workspace. '
                f'board=({board_x:.4f}, {board_y:.4f}) '
                f'local=({local_x:.4f}, {local_y:.4f})',
                params,
            )
            return None
        if not self._inside_two_link_bounds(local_x, local_y):
            self._enter_error(
                'Arm pose target rejected: outside arm reach bounds. '
                f'board=({board_x:.4f}, {board_y:.4f}) '
                f'local=({local_x:.4f}, {local_y:.4f})',
                params,
            )
            return None

        pair = self._best_shoulder_pair(local_x, local_y, params.ik_verify_tol)
        if pair is None:
            self._enter_error(
                'Arm pose target rejected: no verified shoulder solution exists.',
                params,
            )
            return None

        return ActiveGoal(
            board_x=board_x,
            board_y=board_y,
            body_x=body_x,
            body_y=body_y,
            local_x=local_x,
            local_y=local_y,
            theta_l=pair[0],
            theta_r=pair[1],
        )

    def _on_timer(self) -> None:
        params = self._read_params()

        if not params.enabled:
            if self._enabled_last:
                self._clear_active_goal()
                self._publish_pen(params.pen_up_pos)
                self._set_status(self._STATUS_IDLE)
            self._enabled_last = False
            return

        self._enabled_last = True

        if self._active_goal is None:
            if self._status != self._STATUS_ERROR:
                self._set_status(self._STATUS_IDLE)
            return

        if not self._pose_fresh(params.pose_timeout_sec):
            self._enter_error(
                'Arm pose controller error: robot pose went stale during active goal.',
                params,
            )
            return
        if not self._pen_pose_fresh(params.pen_pose_timeout_sec):
            self._enter_error(
                'Arm pose controller error: pen pose went stale during active goal.',
                params,
            )
            return

        self._publish_pen(params.pen_up_pos)
        self._publish_shoulder_targets(
            self._active_goal.theta_l,
            self._active_goal.theta_r,
        )

        assert self._pen_x is not None and self._pen_y is not None
        distance = math.hypot(
            self._pen_x - self._active_goal.board_x,
            self._pen_y - self._active_goal.board_y,
        )
        if distance <= params.target_reached_tol:
            self._set_status(self._STATUS_REACHED)
        else:
            self._set_status(self._STATUS_MOVING)


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
