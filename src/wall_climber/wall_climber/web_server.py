"""ROS-managed HTTP server for the local wall_climber web UI.

This node serves the static web frontend on port 8080 and shuts down
with the rest of the ROS launch system. It optionally opens the UI in a
browser as a best-effort convenience.
"""

import os
import threading
import webbrowser
import http.server
import functools

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


class WebServerNode(Node):
    def __init__(self):
        super().__init__('web_ui_server')
        self._port = int(self.declare_parameter('port', 8080).value)

        pkg_dir = get_package_share_directory('wall_climber')
        web_dir = os.path.join(pkg_dir, 'web')

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=web_dir,
        )
        self._httpd = http.server.HTTPServer(('0.0.0.0', self._port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        url = f'http://localhost:{self._port}'
        self.get_logger().info(f'Web UI serving at {url}')

        # Open browser (non-blocking, best-effort)
        try:
            webbrowser.open(url)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = WebServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._httpd.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
