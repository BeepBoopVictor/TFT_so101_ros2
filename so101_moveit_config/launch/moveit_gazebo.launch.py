import os
import yaml

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def _load_yaml(path: str) -> dict:
    """Load a YAML file safely and return an empty dict if it is missing/invalid."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def generate_launch_description():
    # ============================================================
    # 1) Build MoveIt configuration (URDF, SRDF, pipelines, limits…)
    # ============================================================
    moveit_config = (
        MoveItConfigsBuilder("so101_new_calib", package_name="so101_moveit_config")
        .to_moveit_configs()
    )

    # ============================================================
    # 2) Resolve package paths (robust, avoids FindPackageShare quirks)
    # ============================================================
    pkg_gazebo_share = get_package_share_directory("so101_gazebo")
    pkg_moveit_share = get_package_share_directory("so101_moveit_config")

    # RViz config file (preconfigured MoveIt panels, planning scene, etc.)
    rviz_config_path = os.path.join(pkg_moveit_share, "config", "moveit.rviz")

    # ============================================================
    # 3) Load extra MoveIt YAML configs (controllers + Pilz cartesian limits)
    # ============================================================
    moveit_controllers_path = os.path.join(pkg_moveit_share, "config", "moveit_controllers.yaml")
    moveit_controllers = _load_yaml(moveit_controllers_path)

    # Pilz Cartesian limits configuration (used by pilz_industrial_motion_planner)
    pilz_limits_path = os.path.join(pkg_moveit_share, "config", "pilz_cartesian_limits.yaml")
    pilz_limits_yaml = _load_yaml(pilz_limits_path)

    # MoveIt expects these limits under robot_description_planning

    joint_limits_path = os.path.join(pkg_moveit_share, "config", "joint_limits.yaml")
    joint_limits_yaml = _load_yaml(joint_limits_path)

    robot_description_planning = {
        "robot_description_planning": {
            # Joint limits: vienen como top-level {default_..., joint_limits:{...}}
            **joint_limits_yaml,
            # Pilz: lo metemos como cartesian_limits:{...}
            "cartesian_limits": pilz_limits_yaml.get("cartesian_limits", {}),
        }
    }

    moveit_params = {}
    moveit_params.update(moveit_config.robot_description)
    moveit_params.update(moveit_config.robot_description_semantic)
    moveit_params.update(moveit_config.robot_description_kinematics)
    moveit_params.update(moveit_config.planning_pipelines)
    moveit_params.update(moveit_controllers)
    if "robot_description_planning" in moveit_params:
        del moveit_params["robot_description_planning"]
    moveit_params.update(robot_description_planning)
    moveit_params.update({"use_sim_time": True})


    # ============================================================
    # 4) Gazebo simulation (includes your Gazebo launch)
    # ============================================================
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_share, "launch", "gazebo.launch.py")
        )
    )

    # ============================================================
    # 5) ros2_control controller spawners (optional but typical in sim)
    # ============================================================
    # joint_state_broadcaster_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     output="screen",
    #     arguments=[
    #         "joint_state_broadcaster",
    #         "--controller-manager", "/controller_manager",
    #         "--controller-manager-timeout", "60",
    #         "--switch-timeout", "60",
    #     ],
    # )

    # arm_controller_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     output="screen",
    #     arguments=[
    #         "arm_controller",
    #         "--controller-manager", "/controller_manager",
    #         "--controller-manager-timeout", "60",
    #         "--switch-timeout", "60",
    #     ],
    # )

    # ============================================================
    # 6) MoveIt core: move_group (planning + execution backend)
    # ============================================================
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            # Robot model and planning configuration from MoveItConfigsBuilder
            moveit_params
        ],
    )

    # ============================================================
    # 7) RViz: visualization + interactive planning (MotionPlanning panel)
    # ============================================================
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
        parameters=[
            # Same MoveIt params so RViz can display the robot & planning scene
            # moveit_params
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": True},
        ],
    )

    # ============================================================
    # 8) Launch order
    # ============================================================
    return LaunchDescription(
        [
            gazebo_launch,
            # joint_state_broadcaster_spawner,
            # arm_controller_spawner,
            run_move_group_node,
            rviz_node,
        ]
    )
