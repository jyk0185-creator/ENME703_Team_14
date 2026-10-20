import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):

    robot_namespace = LaunchConfiguration("namespace").perform(context)

    pkg_dir = get_package_share_directory("localization_server_TEAM_14")

    nav2_yaml = os.path.join(pkg_dir, "config", f"{robot_namespace}_amcl_config.yaml")

    map_yaml = os.path.join(pkg_dir, "config", "turtlebot_area.yaml") #turtlebot_area.yaml is the name of the map in the folder /config !!!!
   
    return [
        Node(
            namespace=robot_namespace,
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'yaml_filename': map_yaml}]
        ),
        Node(
            namespace=robot_namespace,
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[nav2_yaml]
        ),
        Node(
            namespace=robot_namespace,
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[
                {"use_sim_time": False},
                {"autostart": True},
                {"bond_timeout": 0.0},
                {"node_names": ['map_server', 'amcl']}
            ]
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "namespace",
            default_value="tb407",
            description="Robot namespace"),
            OpaqueFunction(function=launch_setup)
    ])