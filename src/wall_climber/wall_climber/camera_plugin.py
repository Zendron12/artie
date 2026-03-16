"""Vision camera plugin — compressed camera only, no ArUco.

Runs inside the vision_camera Webots Robot controller. Every step():
  1. Grabs the BGRA frame from the Webots camera device via getImage()
  2. Wraps the frame with np.frombuffer (zero-copy)
  3. JPEG-encodes the BGR image
  4. Publishes sensor_msgs/CompressedImage on
     /vision_camera/image_raw/compressed

No ArUco detection or pose publishing is performed.
"""

import numpy as np
import cv2
import rclpy
from sensor_msgs.msg import CompressedImage


class CameraPlugin:
    CAMERA_NODE_TYPE = 4
    _IMAGE_EVERY = 2  # publish compressed image every N-th step (~15 fps)
    _JPEG_QUALITY = 50

    def init(self, webots_node, properties):
        self.__camera = None
        self.__img_pub = None
        self.__frame_id = 'camera_optical_frame'

        self.__robot = webots_node.robot
        self.__timestep = int(self.__robot.getBasicTimeStep())

        if not rclpy.ok():
            rclpy.init(args=None)
        self.__ros_node = rclpy.create_node('vision_camera_driver')
        self.__log = self.__ros_node.get_logger()

        target_name = properties.get('camera_name', 'camera_link')
        self.__camera = self.__robot.getDevice(target_name)

        if self.__camera is None:
            self.__log.warn(
                f"Device '{target_name}' not found. "
                "Searching for any Camera device..."
            )
            num_devices = self.__robot.getNumberOfDevices()
            for i in range(num_devices):
                device = self.__robot.getDeviceByIndex(i)
                if device.getNodeType() == self.CAMERA_NODE_TYPE:
                    self.__camera = device
                    self.__log.info(
                        f"Found fallback camera: '{device.getName()}'"
                    )
                    break

        if self.__camera is None:
            self.__log.error('No camera device found.')
            return

        self.__camera.enable(self.__timestep)
        self.__width = self.__camera.getWidth()
        self.__height = self.__camera.getHeight()
        self.__frame_id = properties.get('frame_id', 'camera_optical_frame')

        self.__img_pub = self.__ros_node.create_publisher(
            CompressedImage, '/vision_camera/image_raw/compressed', 1
        )
        self.__step_count = 0

        self.__log.info(
            f'Compressed camera plugin active  '
            f'{self.__width}x{self.__height}  '
            f'publish every {self._IMAGE_EVERY} steps  '
            f'jpeg_quality={self._JPEG_QUALITY}'
        )

    def step(self):
        if self.__camera is None or self.__img_pub is None:
            return

        rclpy.spin_once(self.__ros_node, timeout_sec=0)

        self.__step_count += 1
        if self.__step_count % self._IMAGE_EVERY != 0:
            return

        raw = self.__camera.getImage()
        if raw is None:
            return

        bgra = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.__height, self.__width, 4)
        )
        bgr = bgra[:, :, :3]
        ok, jpeg = cv2.imencode(
            '.jpg',
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self._JPEG_QUALITY],
        )
        if not ok:
            return

        img_msg = CompressedImage()
        img_msg.header.stamp = self.__ros_node.get_clock().now().to_msg()
        img_msg.header.frame_id = self.__frame_id
        img_msg.format = 'jpeg'
        img_msg.data = jpeg.tobytes()
        self.__img_pub.publish(img_msg)
