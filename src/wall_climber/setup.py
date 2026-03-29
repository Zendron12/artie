from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wall_climber'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Ament index resource marker (REQUIRED for ament_python packages)
        ('share/ament_index/resource_index/packages',
         [os.path.join('resource', package_name)]),

        # Package manifest
        (os.path.join('share', package_name), ['package.xml']),

        # Launch / URDF / Worlds
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),

        # RViz configs (recommended location)
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),

        # Web UI
        (os.path.join('share', package_name, 'web'), glob('web/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hisham',
    maintainer_email='2144934@std.hu.edu.jo',
    description='Wall Climbing Robot Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'urdf_spawner = wall_climber.urdf_spawner:main',
            'web_server = wall_climber.web_server:main',
            'pose_correction_controller = wall_climber.pose_correction_controller:main',
            'line_demo_controller = wall_climber.line_demo_controller:main',
            'stroke_executor = wall_climber.stroke_executor:main',
            'arm_pose_controller = wall_climber.arm_pose_controller:main',
        ],
    },
)
