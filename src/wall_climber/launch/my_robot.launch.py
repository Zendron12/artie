"""Main system launch file for the wall climber Webots simulation.

This launch description brings up:
- the Webots world and ROS 2 supervisor
- the camera robot and its driver
- the wall-climbing robot and its driver
- the magnetic supervisor helper robot
- the web UI bridge/services
- higher-level motion and drawing controllers

The launch order is intentionally staged with TimerAction delays so the
simulation, supervisor, and spawned robots come up in a predictable order.
"""

import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


def generate_launch_description():
    package_name = 'wall_climber'
    pkg_dir = get_package_share_directory(package_name)
    webots_prefix = LaunchConfiguration('webots_prefix')

    # ------------------------------------------------------------------
    # Resolve package-local resources
    # ------------------------------------------------------------------
    world_path = os.path.join(pkg_dir, "worlds", "wall_world.wbt")
    camera_xacro_path = os.path.join(pkg_dir, "urdf", "vision_camera.xacro")
    climber_xacro_path = os.path.join(pkg_dir, "urdf", "my_robot.urdf.xacro")
    supervisor_xacro_path = os.path.join(pkg_dir, "urdf", "magnetic_supervisor.urdf.xacro")

    # ------------------------------------------------------------------
    # Expand Xacro descriptions into XML strings for robot_state_publisher
    # and for the generic URDF spawner nodes.
    # ------------------------------------------------------------------
    camera_description = ParameterValue(
        Command(['xacro ', camera_xacro_path]),
        value_type=str
    )
    climber_description = ParameterValue(
        Command(['xacro ', climber_xacro_path]),
        value_type=str
    )

    # ------------------------------------------------------------------
    # Start the Webots world and enable the ROS 2 supervisor bridge.
    # ------------------------------------------------------------------
    webots = WebotsLauncher(
        world=world_path,
        mode='realtime',
        ros2_supervisor=True,
        prefix=webots_prefix,
    )

    # ------------------------------------------------------------------
    # Camera robot pipeline
    # ------------------------------------------------------------------

    # Publish the camera URDF under its own namespace to keep TF and topics
    # separated from the wall-climbing robot.
    camera_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='vision_camera',
        parameters=[{'robot_description': camera_description}]
    )

    # Spawn the camera robot at its fixed observation position.
    camera_spawner = Node(
        package=package_name,
        executable='urdf_spawner',
        name='camera_spawner',
        output='screen',
        parameters=[
            {'robot_description': camera_description},
            {'robot_name': 'vision_camera'},
            {'spawn_translation': '0 -3.5 0'},
            {'spawn_rotation': '0 0 1 1.57'}
        ]
    )

    # Webots controller for the spawned camera robot.
    vision_camera_driver = WebotsController(
        robot_name='vision_camera',
        parameters=[
            {'robot_description': camera_xacro_path},
            {'use_sim_time': True}
        ],
        respawn=True
    )

    # ------------------------------------------------------------------
    # Wall-climbing robot pipeline
    # ------------------------------------------------------------------

    # Publish the wall climber URDF in its own namespace.
    climber_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='wall_climber',
        parameters=[{'robot_description': climber_description}]
    )

    # Spawn the climber on the whiteboard at the calibrated start pose.
    climber_spawner = Node(
        package=package_name,
        executable='urdf_spawner',
        name='climber_spawner',
        output='screen',
        parameters=[
            {'robot_description': climber_description},
            {'robot_name': 'wall_climber'},
            {'spawn_translation': '0 2.405 1.80'},
            {'spawn_rotation': '1 0 0 1.57'}
        ]
    )

    # Webots controller for the climber, including drive and temporary
    # keyboard/manual testing plugins.
    wall_climber_driver = WebotsController(
        robot_name='wall_climber',
        parameters=[
            {'robot_description': climber_xacro_path},
            {'use_sim_time': True}
        ],
        respawn=True
    )

    # ------------------------------------------------------------------
    # Magnetic supervisor helper robot
    # ------------------------------------------------------------------
    magnetic_supervisor_driver = WebotsController(
        robot_name='magnetic_supervisor',
        parameters=[
            {'robot_description': supervisor_xacro_path},
            {'use_sim_time': True}
        ],
        respawn=True
    )

    # ------------------------------------------------------------------
    # Launch sequence
    # ------------------------------------------------------------------
    # The ordered TimerAction blocks below avoid race conditions between:
    # - Webots startup
    # - supervisor availability
    # - robot spawning
    # - controller attachment
    # - higher-level ROS nodes that depend on robot/board state
    return LaunchDescription([
        DeclareLaunchArgument(
            'webots_prefix',
            default_value='',
            description=(
                'Optional command prefix for the Webots process itself. '
                'Example: webots_prefix:=mangohud'
            ),
        ),
        webots,
        webots._supervisor,

        # 1. Start both robot_state_publisher instances.
        camera_rsp,
        climber_rsp,

        # 2. Spawn the camera first, then attach its Webots driver.
        camera_spawner,
        launch.actions.TimerAction(
            period=2.0,
            actions=[vision_camera_driver]
        ),

        # 3. Start the magnetic supervisor helper already present in the world.
        launch.actions.TimerAction(
            period=1.0,
            actions=[magnetic_supervisor_driver]
        ),

        # 4. Spawn the wall climber after Webots and the supervisor are ready.
        launch.actions.TimerAction(
            period=5.0,
            actions=[climber_spawner]
        ),

        # 5. Attach the climber Webots driver after the robot exists.
        launch.actions.TimerAction(
            period=7.0,
            actions=[wall_climber_driver]
        ),

        # 6. Start rosbridge for the browser-based UI and tools.
        launch.actions.TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='rosbridge_server',
                    executable='rosbridge_websocket',
                    name='rosbridge_websocket',
                    output='screen',
                ),
            ]
        ),

        # 7. Serve the local web UI on port 8080.
        launch.actions.TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='wall_climber',
                    executable='web_server',
                    name='web_ui_server',
                    output='screen',
                ),
            ]
        ),

        # 8. Optional pose-hold controller for body drift correction.
        launch.actions.TimerAction(
            period=9.0,
            actions=[
                Node(
                    package='wall_climber',
                    executable='pose_correction_controller',
                    name='pose_correction_controller',
                    output='screen',
                    parameters=[
                        {'enabled': False},
                        {'forward_cmd': 0.10},
                        {'k_y': 0.35},
                        {'k_theta': 0.35},
                        {'max_lateral_cmd': 0.08},
                        {'max_angular_cmd': 0.08},
                        {'omega_sign': -1.0},
                        {'auto_capture_y_reference': True},
                        {'auto_capture_theta_reference': False},
                        {'target_theta': 0.0},
                    ],
                ),
            ]
        ),

        # 9. Optional two-line board-aware demo controller.
        #    Publishes body motion and pen targets for the legacy demo path.
        launch.actions.TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='wall_climber',
                    executable='line_demo_controller',
                    name='line_demo_controller',
                    output='screen',
                    parameters=[
                        {'enabled': False},
                        {'start_margin_x': 0.20},
                        {'end_margin_x': 0.20},
                        {'line_spacing': 0.18},
                        {'draw_speed': 0.45},
                        {'reposition_speed': 0.80},
                        {'target_theta': 0.0},
                        {'k_y': 0.75},
                        {'k_theta': 0.60},
                        {'omega_sign': -1.0},
                        {'max_lateral_cmd': 0.30},
                        {'max_angular_cmd': 0.22},
                        {'contact_required_for_drawing': True},
                        {'contact_gap_min': -0.0018},
                        {'contact_gap_max': 0.0018},
                        {'pen_settle_cycles': 4},
                        {'lost_contact_cycles_before_reprobe': 8},
                        {'lost_contact_gap_threshold': 0.004},
                        {'draw_pen_extra_depth': 0.0},
                        {'draw_pen_recover_step': 0.0},
                        {'pen_up_pos': 0.018},
                        {'pen_clear_gap': 0.004},
                        {'pen_lift_timeout_sec': 1.5},
                        {'pen_down_min_pos': -0.010},
                        {'pen_down_max_pos': -0.030},
                        {'pen_probe_step': 0.0005},
                        {'pen_contact_timeout_sec': 1.5},
                    ],
                ),
            ]
        ),

        # 10. Generic JSON stroke executor for body-based drawing.
        launch.actions.TimerAction(
            period=11.0,
            actions=[
                Node(
                    package='wall_climber',
                    executable='stroke_executor',
                    name='stroke_executor',
                    output='screen',
                    parameters=[
                        {'enabled': False},
                        {'draw_speed': 0.45},
                        {'reposition_speed': 0.80},
                        {'target_theta': 0.0},
                        {'k_y': 0.75},
                        {'k_theta': 0.60},
                        {'omega_sign': -1.0},
                        {'max_lateral_cmd': 0.30},
                        {'max_angular_cmd': 0.22},
                        {'pos_tol_x': 0.004},
                        {'pos_tol_y': 0.004},
                        {'contact_required_for_drawing': True},
                        {'contact_gap_min': -0.0018},
                        {'contact_gap_max': 0.0018},
                        {'pen_settle_cycles': 12},
                        {'corner_settle_cycles': 9},
                        {'draw_start_delay_cycles': 0},
                        {'lost_contact_cycles_before_reprobe': 24},
                        {'lost_contact_gap_threshold': 0.004},
                        {'draw_pen_extra_depth': 0.0},
                        {'draw_pen_recover_step': 0.0},
                        {'pen_up_pos': 0.018},
                        {'pen_clear_gap': 0.004},
                        {'pen_lift_timeout_sec': 1.5},
                        {'pen_down_min_pos': -0.010},
                        {'pen_down_max_pos': -0.030},
                        {'pen_probe_step': 0.0005},
                        {'pen_contact_timeout_sec': 1.5},
                    ],
                ),
            ]
        ),

        # 11. Arm point-to-point pose controller.
        #     This phase drives only the pantograph arm and keeps the pen up.
        launch.actions.TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='wall_climber',
                    executable='arm_pose_controller',
                    name='arm_pose_controller',
                    output='screen',
                    parameters=[
                        {'enabled': False},
                        {'pose_timeout_sec': 0.5},
                        {'pen_pose_timeout_sec': 0.5},
                        {'pen_up_pos': 0.018},
                        {'target_reached_tol': 0.010},
                        {'ik_verify_tol': 0.002},
                        {'local_x_min': -0.09},
                        {'local_x_max': 0.09},
                        {'local_y_min': 0.22},
                        {'local_y_max': 0.32},
                    ],
                ),
            ]
        ),

        # Clean shutdown: when Webots exits, bring down the whole launch tree.
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])
