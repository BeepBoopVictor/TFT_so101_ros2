import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch.conditions import IfCondition

from ament_index_python.packages import get_package_share_directory


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

    # Make Gazebo able to find package resources (meshes, textures, etc.)
    _extend_gz_resource_path(os.path.join(pkg_description, ".."))

    # ============================================================
    # 2) Build robot_description from Xacro
    # ============================================================
    xacro_file = os.path.join(pkg_description, "urdf", "so101_new_calib.urdf.xacro")

    robot_description = ParameterValue( Command(["xacro ", xacro_file, " mode:=gazebo"]), value_type=str, )

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
    pkg_gazebo_share = get_package_share_directory("so101_gazebo")
    world_path = os.path.join(pkg_gazebo_share, "worlds", "so101_with_camera.sdf")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare("ros_gz_sim").find("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={
        	"gz_args": f"-r -s {world_path}"
        }.items(),
        # launch_arguments={"gz_args": f"-r empty.sdf"}.items(),
    )

    # Spawn the robot entity in Gazebo
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "so101_new_calib",
        ],
    )

    # Spawn the camera (CORREGIDO: fuera del bloque de spawn_robot)
    spawn_camera = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-file", os.path.join(pkg_gazebo_share, "worlds", "external_camera.sdf"),
            "-name", "camera",
        ],
    )

    # ============================================================
    # 5) BRIDGE: Gazebo <-> ROS 2
    # ============================================================
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
    )

    # ============================================================
    # 6) ros2_control: spawn controllers (ordered)
    # ============================================================
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

    load_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager-timeout", "60"],
        output="screen",
    )

    load_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager-timeout", "60"],
        output="screen",
    )

    start_arm_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_arm_controller],
        )
    )

    start_gripper_after_arm = RegisterEventHandler(
        OnProcessExit(
            target_action=load_arm_controller,
            on_exit=[load_gripper_controller],
        )
    )
    
    
    
    
    # Argumento para activar/desactivar el visor
    use_rqt = LaunchConfiguration("use_rqt")

    declare_use_rqt = DeclareLaunchArgument(
        "use_rqt",
        default_value="true",
        description="Launch rqt_image_view to display /camera/image_raw",
    )

    # Nodo rqt_image_view (solo si use_rqt:=true)
    rqt_image_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_image_view",
        output="screen",
        arguments=["/camera/image_raw"],
        condition=IfCondition(use_rqt),
    )    

    # ============================================================
    # 7) Launch description
    # ============================================================
    return LaunchDescription(
        [
            declare_use_rqt,
            
            robot_state_publisher,
            gazebo,
            spawn_robot,
            spawn_camera,
            gz_bridge,
            start_jsb_after_spawn,
            start_arm_after_jsb,
            start_gripper_after_arm,
            
            rqt_image_view,
        ]
    )
