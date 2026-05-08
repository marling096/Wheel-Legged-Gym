from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PACKAGE_NAME = 'swpu2026'


def generate_launch_description():
    share_dir = Path(get_package_share_directory(PACKAGE_NAME))
    urdf_path = share_dir / 'urdf' / 'swpu2026.urdf'
    rviz_config_path = share_dir / 'rviz' / 'display.rviz'

    robot_description = urdf_path.read_text(encoding='utf-8')

    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            choices=['true', 'false'],
            description='Launch joint_state_publisher_gui when true.',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            choices=['true', 'false'],
            description='Launch rviz2 when true.',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=str(rviz_config_path),
            description='Absolute path to the rviz2 config file.',
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            condition=UnlessCondition(gui),
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            condition=IfCondition(gui),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(rviz),
            output='screen',
        ),
    ])
