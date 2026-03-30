"""Helper node that requests URDF robot spawning from Ros2Supervisor.

This node packages a robot description and spawn pose into a
webots_ros2_msgs/SpawnUrdfRobot request and waits for the supervisor
service to become available before sending it.
"""

import rclpy
from rclpy.node import Node
from webots_ros2_msgs.srv import SpawnUrdfRobot
from webots_ros2_msgs.msg import UrdfRobot


class URDFSpawnerNode(Node):
    def __init__(self):
        super().__init__('urdf_spawner')

        # Service path provided by Ros2Supervisor in ROS 2 Humble.
        self.client = self.create_client(SpawnUrdfRobot, '/Ros2Supervisor/spawn_urdf_robot')

        self.declare_parameter('robot_description', '')
        self.declare_parameter('robot_name', 'vision_camera')
        self.declare_parameter('spawn_translation', '0 -3.5 0')
        self.declare_parameter('spawn_rotation', '0 0 1 1.5708')

        max_wait_seconds = 60
        elapsed_seconds = 0
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /Ros2Supervisor/spawn_urdf_robot service...')
            elapsed_seconds += 1
            if elapsed_seconds >= max_wait_seconds:
                raise RuntimeError(
                    'Timed out waiting for /Ros2Supervisor/spawn_urdf_robot service.'
                )

        urdf_content = self.get_parameter('robot_description').value
        robot_name = self.get_parameter('robot_name').value
        spawn_translation = self.get_parameter('spawn_translation').value
        spawn_rotation = self.get_parameter('spawn_rotation').value

        if not urdf_content:
            raise ValueError('Parameter robot_description is empty.')

        request = SpawnUrdfRobot.Request()
        robot = UrdfRobot()
        robot.name = robot_name
        robot.robot_description = urdf_content
        robot.translation = spawn_translation
        robot.rotation = spawn_rotation
        request.robot = robot

        self.get_logger().info(
            f'Sending spawn request for "{robot_name}" at {spawn_translation}'
            f' with rotation {spawn_rotation}.'
        )
        self.future = self.client.call_async(request)


def main():
    rclpy.init()
    node = None
    try:
        node = URDFSpawnerNode()
        rclpy.spin_until_future_complete(node, node.future)

        response = node.future.result()
        if response and response.success:
            node.get_logger().info('Spawn request completed successfully.')
        else:
            node.get_logger().error('Spawn request failed according to Ros2Supervisor response.')
    except Exception as error:
        if node is not None:
            node.get_logger().error(f'URDF spawner failed: {error}')
        else:
            print(f'URDF spawner failed before node creation: {error}')
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
