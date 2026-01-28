import os
import yaml
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory # Importante para rutas limpias


def generate_launch_description():
    # 1. Cargar la configuración de MoveIt
    moveit_config = MoveItConfigsBuilder("so101_new_calib", package_name="so101_moveit_config").to_moveit_configs()

    # 2. Obtener la ruta del paquete de Gazebo de forma segura
    # Esto evita el error de la 'tupla' al no usar .find() dentro de IncludeLaunchDescription
    pkg_gazebo_share = get_package_share_directory('so101_gazebo')
    
    pkg_moveit_share = get_package_share_directory("so101_moveit_config")
    pilz_limits_path = os.path.join(pkg_moveit_share, "config", "pilz_cartesian_limits.yaml")
    
    with open(pilz_limits_path, "r") as f:
    	pilz_limits = yaml.safe_load(f)
    
    robot_description_planning_pilz = {"robot_description_planning": pilz_limits}
    
    rviz_config_path = os.path.join(pkg_moveit_share, "config", "moveit.rviz")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_share, 'launch', 'gazebo.launch.py')
        )
    )
    
    moveit_controllers_path = os.path.join(pkg_moveit_share, "config", "moveit_controllers.yaml")
    moveit_controllers = yaml.safe_load(open(moveit_controllers_path, "r"))
    
    robot_description_planning_pilz = {
      "robot_description_planning": {
        "cartesian_limits": {
          "max_trans_vel": 1.0,
          "max_trans_acc": 2.25,
          "max_trans_dec": -5.0,
          "max_rot_vel": 1.57,
        }
      }
    }

    # 3. Nodo Move Group (el motor de planificación)
    #run_move_group_node = Node(
    #    package="moveit_ros_move_group",
    #    executable="move_group",
    #    output="screen",
    #    # Fusionamos los diccionarios de parámetros correctamente
    #    parameters=[
    #        moveit_config.to_dict(),
    #        {'use_sim_time': True}
    #    ],
    #)
    
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "60",
            "--switch-timeout", "60",
        ],
        output="screen",
    )
    
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "60",
            "--switch-timeout", "60",
        ],
        output="screen",
    )


    
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            moveit_controllers,
            robot_description_planning_pilz,
            {"use_sim_time": True},
        ],
    )

    # 4. RViz para visualizar y mover el brazo
    #rviz_node = Node(
    #    package="rviz2",
    #    executable="rviz2",
    #    output="screen",
    #    parameters=[
    #        moveit_config.to_dict(),
    #        {'use_sim_time': True}
    #    ],
    #)
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_controllers,
            moveit_config.joint_limits,
            robot_description_planning_pilz,
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([
        gazebo_launch,
        run_move_group_node,
        rviz_node
    ])
