"""Vision camera plugin — local ArUco detection, no raw image publishing.

Runs inside the vision_camera Webots Robot controller.  Every step():
  1. Grabs the BGRA frame from the Webots camera device via getImage()
  2. Converts to grayscale in-place using np.frombuffer (zero-copy)
  3. Detects ArUco markers (DICT_4X4_50)
  4. Estimates each marker's 6-DoF pose with cv2.solvePnP using
     camera intrinsics derived from the known FOV and resolution
  5. Publishes results as geometry_msgs/PoseArray on
     /vision_camera/aruco_poses  (lightweight, ~100 bytes per marker)

No raw image is ever serialised over DDS → simulation stays real-time.
"""

import math
import numpy as np
import cv2
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header


def _rodrigues_to_quat(rvec):
    """Convert a Rodrigues rotation vector to (w, x, y, z) quaternion."""
    R, _ = cv2.Rodrigues(rvec)
    # transforms3d is not always available; inline the conversion.
    # Quaternion from 3×3 rotation matrix (Shepperd's method).
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


class CameraPlugin:
    CAMERA_NODE_TYPE = 4

    # ArUco config
    _ARUCO_DICT    = cv2.aruco.DICT_4X4_50
    _MARKER_SIZE   = 0.10    # 10 cm markers (tune to match your printed markers)
    _DETECT_EVERY  = 4       # detect every N-th step (skip frames for speed)
    _IMAGE_EVERY   = 2       # publish compressed image every N-th step (~15 fps)

    def init(self, webots_node, properties):
        self.__camera = None
        self.__pub = None
        self.__frame_id = 'camera_optical_frame'

        self.__robot = webots_node.robot
        self.__timestep = int(self.__robot.getBasicTimeStep())

        # ---- ROS 2 node ----
        if not rclpy.ok():
            rclpy.init(args=None)
        self.__ros_node = rclpy.create_node('vision_camera_driver')
        self.__log = self.__ros_node.get_logger()

        # ---- camera device ----
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
        self.__width  = self.__camera.getWidth()
        self.__height = self.__camera.getHeight()

        # ---- camera intrinsics from FOV + resolution ----
        fov = float(properties.get('fov', '1.2'))  # horizontal FOV in rad
        fx = (self.__width / 2.0) / math.tan(fov / 2.0)
        fy = fx  # square pixels
        cx = self.__width  / 2.0
        cy = self.__height / 2.0

        self.__cam_matrix = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1.0],
        ], dtype=np.float64)
        self.__dist_coeffs = np.zeros(5, dtype=np.float64)  # no distortion

        # ---- ArUco detector (OpenCV 4.5 legacy API) ----
        self.__aruco_dict = cv2.aruco.Dictionary_get(self._ARUCO_DICT)
        self.__aruco_params = cv2.aruco.DetectorParameters_create()
        # Speed tweaks
        self.__aruco_params.adaptiveThreshWinSizeMin  = 3
        self.__aruco_params.adaptiveThreshWinSizeMax  = 23
        self.__aruco_params.adaptiveThreshWinSizeStep = 10

        # ---- publishers ----
        self.__frame_id = properties.get('frame_id', 'camera_optical_frame')
        self.__pub = self.__ros_node.create_publisher(
            PoseArray, '/vision_camera/aruco_poses', 1
        )
        self.__img_pub = self.__ros_node.create_publisher(
            CompressedImage, '/vision_camera/image_raw/compressed', 1
        )
        self.__step_count = 0

        self.__log.info(
            f'ArUco camera plugin active  '
            f'{self.__width}x{self.__height}  '
            f'FOV={fov:.2f} rad  fx={fx:.1f}  '
            f'dict=4X4_50  marker={self._MARKER_SIZE}m  '
            f'detect every {self._DETECT_EVERY} steps'
        )

    # ------------------------------------------------------------------
    def step(self):
        if self.__camera is None or self.__pub is None:
            return

        rclpy.spin_once(self.__ros_node, timeout_sec=0)

        self.__step_count += 1

        do_aruco = (self.__step_count % self._DETECT_EVERY == 0)
        do_image = (self.__step_count % self._IMAGE_EVERY == 0)

        if not do_aruco and not do_image:
            return

        # ---- grab frame (zero-copy into numpy) ----
        raw = self.__camera.getImage()
        if raw is None:
            return

        # Webots gives BGRA as bytes; wrap as uint8 array
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.__height, self.__width, 4)
        )

        # ---- publish compressed JPEG for Foxglove (low rate) ----
        if do_image:
            bgr = bgra[:, :, :3]  # drop alpha
            ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if ok:
                img_msg = CompressedImage()
                img_msg.header.stamp = self.__ros_node.get_clock().now().to_msg()
                img_msg.header.frame_id = self.__frame_id
                img_msg.format = 'jpeg'
                img_msg.data = jpeg.tobytes()
                self.__img_pub.publish(img_msg)

        # ---- ArUco detection ----
        if not do_aruco:
            return

        gray = bgra[:, :, 1]  # green channel ≈ luminance

        # ---- detect markers (legacy API) ----
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.__aruco_dict, parameters=self.__aruco_params
        )

        if ids is None or len(ids) == 0:
            return  # nothing found → don't publish empty

        # ---- estimate poses ----
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self._MARKER_SIZE, self.__cam_matrix, self.__dist_coeffs
        )

        # ---- build PoseArray ----
        msg = PoseArray()
        msg.header = Header()
        msg.header.stamp = self.__ros_node.get_clock().now().to_msg()
        msg.header.frame_id = self.__frame_id

        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            tvec = tvecs[i]

            w, x, y, z = _rodrigues_to_quat(rvec)

            pose = Pose()
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])
            pose.orientation.w = w
            pose.orientation.x = x
            pose.orientation.y = y
            pose.orientation.z = z
            msg.poses.append(pose)

        if msg.poses:
            self.__pub.publish(msg)