import os
# from hardships_ros2.substitutions import FindPackageShare
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_description = FindPackageShare('so101_description').find('so101_description')
    pkg_gazebo = FindPackageShare('so101_gazebo').find('so101_gazebo')

    # Ruta al archivo Xacro
    xacro_file = os.path.join(pkg_description, 'urdf', 'so101_new_calib.urdf.xacro')

    # Generar el Robot Description con modo gazebo
    robot_description_content = Command([
        'xacro ', xacro_file, ' mode:=gazebo'
    ])

    # Nodo Robot State Publisher
    node_robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    output='screen',
    parameters=[{
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }]
)

    # Lanzar Gazebo (Mundo vacío)
    # Lanzar Gazebo Sim (Mundo vacío) - SUSTITUIR EL ANTERIOR
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(FindPackageShare('ros_gz_sim').find('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # Spawn del robot en el nuevo Gazebo - SUSTITUIR EL ANTERIOR
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'so101'],
        output='screen'
    )

    # Cargar los controladores
    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    load_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
    )

    return LaunchDescription([
        node_robot_state_publisher,
        gazebo,
        spawn_entity,
        load_joint_state_broadcaster,
        load_arm_controller
    ])
