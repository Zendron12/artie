import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


def generate_launch_description():
    package_name = 'wall_climber'
    pkg_dir = get_package_share_directory(package_name)

    # 1. تحديد المسارات
    world_path = os.path.join(pkg_dir, "worlds", "wall_world.wbt")
    camera_xacro_path = os.path.join(pkg_dir, "urdf", "vision_camera.xacro")
    climber_xacro_path = os.path.join(pkg_dir, "urdf", "my_robot.urdf.xacro")
    supervisor_xacro_path = os.path.join(pkg_dir, "urdf", "magnetic_supervisor.urdf.xacro")

    # 2. تحويل ملفات Xacro إلى نصوص (XML)
    # وصف الكاميرا
    camera_description = ParameterValue(
        Command(['xacro ', camera_xacro_path]),
        value_type=str
    )
    # وصف الروبوت المتسلق
    climber_description = ParameterValue(
        Command(['xacro ', climber_xacro_path]),
        value_type=str
    )

    # 3. تشغيل Webots
    webots = WebotsLauncher(
        world=world_path,
        mode='realtime',
        ros2_supervisor=True
    )

    # ==========================================================
    #   الجزء الأول: الكاميرا (كما هي في الكود الشغال معك)
    # ==========================================================

    # ناشر حالة الكاميرا (مع Namespace عشان نفصلها عن الروبوت الثاني)
    camera_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='vision_camera',  # ضروري جداً للفصل
        parameters=[{'robot_description': camera_description}]
    )

    # سباونر الكاميرا (ينزلها في مكانها الأصلي)
    camera_spawner = Node(
        package=package_name,
        executable='urdf_spawner',
        name='camera_spawner',
        output='screen',
        parameters=[
            {'robot_description': camera_description},
            {'robot_name': 'vision_camera'},
            {'spawn_translation': '0 -3.5 0'},  # مكان الكاميرا
            {'spawn_rotation': '0 0 1 1.57'}
        ]
    )

    # درايفر الكاميرا
    vision_camera_driver = WebotsController(
        robot_name='vision_camera',
        parameters=[
            {'robot_description': camera_xacro_path},
            {'use_sim_time': True}
        ],
        respawn=True
    )

    # ==========================================================
    #   الجزء الثاني: الروبوت المتسلق (الإضافة الجديدة)
    # ==========================================================

    # ناشر حالة المتسلق
    climber_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='wall_climber',  # ضروري جداً للفصل
        parameters=[{'robot_description': climber_description}]
    )

    # سباونر المتسلق (ينزله على اللوح)
    climber_spawner = Node(
        package=package_name,
        executable='urdf_spawner',
        name='climber_spawner',
        output='screen',
        parameters=[
            {'robot_description': climber_description},
            {'robot_name': 'wall_climber'},
            # الإحداثيات الدقيقة للوح
            {'spawn_translation': '0 2.405 1.80'},
            {'spawn_rotation': '1 0 0 1.57'}
        ]
    )

    # درايفر المتسلق (عشان يتحرك لبعدين)
    wall_climber_driver = WebotsController(
        robot_name='wall_climber',
        parameters=[
            {'robot_description': climber_xacro_path},
            {'use_sim_time': True}
        ],
        respawn=True
    )

    # ==========================================================
    #   الجزء الثالث: Supervisor للمغناطيس (روبوت غير مرئي)
    # ==========================================================
    magnetic_supervisor_driver = WebotsController(
        robot_name='magnetic_supervisor',
        parameters=[
            {'robot_description': supervisor_xacro_path},
            {'use_sim_time': True}
        ],
        respawn=True
    )

    # ==========================================================
    #   تجميع الكل وتشغيلهم بتسلسل زمني
    # ==========================================================
    return LaunchDescription([
        webots,
        webots._supervisor,

        # 1. تشغيل الـ State Publishers للروبوتين
        camera_rsp,
        climber_rsp,

        # 2. إنزال الكاميرا أولاً ثم تشغيل درايفرها
        camera_spawner,
        launch.actions.TimerAction(
            period=2.0,
            actions=[vision_camera_driver]
        ),

        # 3. تشغيل درايفر المغناطيس (الروبوت موجود في .wbt)
        launch.actions.TimerAction(
            period=1.0,
            actions=[magnetic_supervisor_driver]
        ),

        # 4. بعد 5 ثواني.. إنزال الروبوت المتسلق
        launch.actions.TimerAction(
            period=5.0,
            actions=[climber_spawner]
        ),

        # 5. تشغيل درايفر المتسلق بعد السبون (بلغن الكيبورد + السويرف)
        launch.actions.TimerAction(
            period=7.0,
            actions=[wall_climber_driver]
        ),

        # 6. rosbridge WebSocket server (for web UI)
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

        # 7. Web UI HTTP server (serves HTML on port 8080)
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

        # 8. Pose-based drift correction publisher (/wall_climber/cmd_vel_auto)
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

        # 9. Two-line board-aware demo controller
        #    (/wall_climber/cmd_vel_auto + /wall_climber/pen_target)
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
                        {'draw_pen_extra_depth': 0.006},
                        {'draw_pen_recover_step': 0.0008},
                        {'pen_up_pos': 0.018},
                        {'pen_clear_gap': 0.004},
                        {'pen_lift_timeout_sec': 1.5},
                        {'pen_down_min_pos': -0.010},
                        {'pen_down_max_pos': -0.030},
                        {'pen_probe_step': 0.0025},
                        {'pen_contact_timeout_sec': 1.5},
                    ],
                ),
            ]
        ),

        # 10. Generic stroke executor
        #     (/wall_climber/stroke_plan -> /wall_climber/cmd_vel_auto + /wall_climber/pen_target)
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
                        {'contact_required_for_drawing': True},
                        {'contact_gap_min': -0.0018},
                        {'contact_gap_max': 0.0018},
                        {'pen_settle_cycles': 4},
                        {'lost_contact_cycles_before_reprobe': 8},
                        {'lost_contact_gap_threshold': 0.004},
                        {'draw_pen_extra_depth': 0.006},
                        {'draw_pen_recover_step': 0.0008},
                        {'pen_up_pos': 0.018},
                        {'pen_clear_gap': 0.004},
                        {'pen_lift_timeout_sec': 1.5},
                        {'pen_down_min_pos': -0.010},
                        {'pen_down_max_pos': -0.030},
                        {'pen_probe_step': 0.0025},
                        {'pen_contact_timeout_sec': 1.5},
                    ],
                ),
            ]
        ),

        # إغلاق نظيف
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])
