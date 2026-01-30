import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _extend_gz_resource_path(path_to_add: str) -> None:
    """
    Append a path to GZ_SIM_RESOURCE_PATH so Gazebo can resolve meshes/materials.
    """
    env_key = "GZ_SIM_RESOURCE_PATH"
    if env_key in os.environ and os.environ[env_key]:
        os.environ[env_key] += os.pathsep + path_to_add
    else:
        os.environ[env_key] = path_to_add


def generate_launch_description():
    # ============================================================
    # 1) Locate package resources
    # ============================================================
    pkg_description = FindPackageShare("so101_description").find("so101_description")
    # NOTE: pkg_gazebo is not used in this launch, keep it only if you plan to use it later.
    # pkg_gazebo = FindPackageShare("so101_gazebo").find("so101_gazebo")

    # Make Gazebo able to find package resources (meshes, textures, etc.)
    _extend_gz_resource_path(os.path.join(pkg_description, ".."))

    # ============================================================
    # 2) Build robot_description from Xacro
    # ============================================================
    xacro_file = os.path.join(pkg_description, "urdf", "so101_new_calib.urdf.xacro")

    # Generate URDF at launch-time (enable Gazebo-specific configuration via mode:=gazebo)
    robot_description = ParameterValue(
        Command(["xacro ", xacro_file, " mode:=gazebo"]),
        value_type=str,
    )

    # ============================================================
    # 3) Core nodes: robot_state_publisher
    # ============================================================
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    # ============================================================
    # 4) Gazebo: start simulator + spawn the robot
    # ============================================================
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("ros_gz_sim").find("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py",
                )
            ]
        ),
        # Equivalent to: gz sim -r empty.sdf
        launch_arguments={"gz_args": "-r empty.sdf"}.items(),
    )

    # Spawn the robot entity in Gazebo from the robot_description source
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",  # Read URDF XML from this source
            "-name", "so101",               # Name of the spawned model in Gazebo
        ],
    )

    # ============================================================
    # 5) Time sync: bridge Gazebo /clock -> ROS 2 /clock
    # ============================================================
    gz_clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
    )

    # ============================================================
    # 6) ros2_control: spawn controllers (ordered)
    # ============================================================
    # First: Joint State Broadcaster so ROS can publish /joint_states
    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager-timeout", "60"],
        output="screen",
    )

    start_jsb_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[load_joint_state_broadcaster],
        )
    )

    # Then: Arm trajectory controller (FollowJointTrajectory)
    load_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager-timeout", "60"],
        output="screen",
    )

    # Start arm_controller only after the spawner for joint_state_broadcaster finishes
    # (spawner exits after successfully loading the controller)
    start_arm_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_arm_controller],
        )
    )

    # ============================================================
    # 7) Launch description (execution order)
    # ============================================================
    return LaunchDescription(
        [
            # Robot description + TF
            robot_state_publisher,

            # Simulator + spawn
            gazebo,
            spawn_robot,

            # Simulation time
            gz_clock_bridge,

            # Controllers
            start_jsb_after_spawn,
            start_arm_after_jsb,
        ]
    )
