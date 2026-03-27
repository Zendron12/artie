"""Additive RViz trail renderer for Artie.

This node mirrors the current Webots drawing semantics into RViz markers
without changing the simulator-side trail rendering.
"""

import json
import math

import rclpy
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker


def _make_point(x: float, y: float, z: float = 0.0) -> Point:
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


class RvizTrailRenderer(Node):
    _TRAIL_TOPIC = '/wall_climber/rviz_trail_markers'
    _MIN_POINT_SPACING = 0.0025
    _TRAIL_WIDTH = 0.012
    _BOARD_WIDTH = 0.008
    _TIP_SIZE = 0.020

    def __init__(self) -> None:
        super().__init__('rviz_trail_renderer')

        self._board = None
        self._marker_frame_id = None
        self._pen_contact = False
        self._drawing_active = False

        self._board_marker_dirty = False

        self._current_stroke_id = 0
        self._current_stroke_points = []
        self._last_draw_xy = None

        self._last_warn_sec = -1e9

        self._marker_pub = self.create_publisher(Marker, self._TRAIL_TOPIC, 10)
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
        self.create_subscription(
            Bool,
            '/wall_climber/drawing_active',
            self._drawing_active_cb,
            10,
        )
        self.create_subscription(
            String,
            '/wall_climber/board_info',
            self._board_info_cb,
            10,
        )

        self._clear_markers()
        self.get_logger().info(
            'RViz trail renderer ready; publishing additive markers on '
            f'{self._TRAIL_TOPIC}.'
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _log_warn_throttled(self, text: str, period_sec: float = 2.0) -> None:
        now = self._now_sec()
        if now - self._last_warn_sec >= period_sec:
            self.get_logger().warn(text)
            self._last_warn_sec = now

    def _clear_markers(self) -> None:
        marker = Marker()
        marker.action = Marker.DELETEALL
        self._marker_pub.publish(marker)

    def _pen_contact_cb(self, msg: Bool) -> None:
        previous = self._pen_contact
        self._pen_contact = bool(msg.data)
        if previous and not self._pen_contact:
            self._finish_current_stroke()

    def _drawing_active_cb(self, msg: Bool) -> None:
        self._drawing_active = bool(msg.data)

    def _board_info_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self._log_warn_throttled(f'Failed to parse /wall_climber/board_info: {exc}')
            return

        needed = (
            'writable_x_min',
            'writable_x_max',
            'writable_y_min',
            'writable_y_max',
        )
        if any(key not in data for key in needed):
            self._log_warn_throttled(
                'board_info JSON missing writable bounds keys; skipping board marker.'
            )
            return

        self._board = data
        self._board_marker_dirty = True
        self._maybe_publish_board_marker()

    def _pen_pose_cb(self, msg: PointStamped) -> None:
        incoming_frame_id = str(msg.header.frame_id).strip()
        if incoming_frame_id:
            if self._marker_frame_id is None:
                self._marker_frame_id = incoming_frame_id
                self._board_marker_dirty = True
            elif incoming_frame_id != self._marker_frame_id:
                self._log_warn_throttled(
                    'Received pen poses with inconsistent frame_id values; '
                    f'keeping the first frame_id={self._marker_frame_id!r}.'
                )
        elif self._marker_frame_id is None:
            return

        self._maybe_publish_board_marker()
        self._publish_tip_marker(msg)

        if not (self._pen_contact and self._drawing_active):
            return

        point = _make_point(msg.point.x, msg.point.y, 0.0)
        point_xy = (point.x, point.y)
        if not self._current_stroke_points:
            # Seed the stroke only. This preserves the existing no-initial-blob
            # semantics instead of stamping a visible point immediately.
            self._current_stroke_points = [point]
            self._last_draw_xy = point_xy
            return

        if self._last_draw_xy is not None:
            distance = math.hypot(
                point_xy[0] - self._last_draw_xy[0],
                point_xy[1] - self._last_draw_xy[1],
            )
            if distance < self._MIN_POINT_SPACING:
                return

        self._current_stroke_points.append(point)
        self._last_draw_xy = point_xy
        self._publish_trail_marker(self._current_stroke_id, self._current_stroke_points)

    def _maybe_publish_board_marker(self) -> None:
        if (
            not self._board_marker_dirty
            or self._board is None
            or self._marker_frame_id is None
        ):
            return

        marker = Marker()
        marker.header.frame_id = self._marker_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'board'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self._BOARD_WIDTH
        marker.color.r = 0.10
        marker.color.g = 0.85
        marker.color.b = 0.95
        marker.color.a = 0.95

        x_min = float(self._board['writable_x_min'])
        x_max = float(self._board['writable_x_max'])
        y_min = float(self._board['writable_y_min'])
        y_max = float(self._board['writable_y_max'])
        marker.points = [
            _make_point(x_min, y_min),
            _make_point(x_max, y_min),
            _make_point(x_max, y_max),
            _make_point(x_min, y_max),
            _make_point(x_min, y_min),
        ]
        self._marker_pub.publish(marker)
        self._board_marker_dirty = False

    def _publish_tip_marker(self, msg: PointStamped) -> None:
        if self._marker_frame_id is None:
            return

        marker = Marker()
        marker.header.frame_id = self._marker_frame_id
        marker.header.stamp = msg.header.stamp
        marker.ns = 'tip'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(msg.point.x)
        marker.pose.position.y = float(msg.point.y)
        marker.pose.position.z = 0.0
        marker.scale.x = self._TIP_SIZE
        marker.scale.y = self._TIP_SIZE
        marker.scale.z = self._TIP_SIZE
        marker.color.r = 0.95
        marker.color.g = 0.15
        marker.color.b = 0.15
        marker.color.a = 0.95
        self._marker_pub.publish(marker)

    def _publish_trail_marker(self, stroke_id: int, points: list[Point]) -> None:
        if self._marker_frame_id is None or len(points) < 2:
            return

        marker = Marker()
        marker.header.frame_id = self._marker_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'trail'
        marker.id = int(stroke_id)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self._TRAIL_WIDTH
        marker.color.r = 0.05
        marker.color.g = 0.05
        marker.color.b = 0.05
        marker.color.a = 1.0
        marker.points = list(points)
        self._marker_pub.publish(marker)

    def _finish_current_stroke(self) -> None:
        if self._current_stroke_points:
            self._current_stroke_id += 1
        self._current_stroke_points = []
        self._last_draw_xy = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RvizTrailRenderer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
