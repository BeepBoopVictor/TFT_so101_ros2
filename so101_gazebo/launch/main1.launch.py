# SO101 - Simulación en Gazebo de mundo completo con robot y cinta transportadora
# Autor: Víctor Gil
# Contacto: 
#   - Correo Institucional: victor.gil107@alu.ulpgc.es
#   - Correo Personal: gbvictor011@protonmail.com
#   - GitHub: BeepBoopVictor
# Este código es el launcher principal para iniciar la simulación completa del robot SO101 en Gazebo, 
# incluyendo el robot, la cinta transportadora y los controladores necesarios para su funcionamiento. 
# Además, se incluye un visor de cámara para facilitar la visualización durante el desarrollo y pruebas.

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch.conditions import IfCondition

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def _extend_gz_resource_path(path_to_add: str) -> None:
    """
        Carga la ruta extendida para encontrar las mallas
    """

    # Extendemos la variable de entorno GZ_SIM_RESOURCE_PATH para incluir la ruta a las mallas del paquete so101_description
    env_key = "GZ_SIM_RESOURCE_PATH"
    if env_key in os.environ and os.environ[env_key]:
        os.environ[env_key] += os.pathsep + path_to_add
    else:
        os.environ[env_key] = path_to_add


def generate_launch_description():
    """
        Inicializa el launcher de la simulación		 
    """

    # --- CONFIGURACIÓN DEL ROBOT Y EL MUNDO ---
    pkg_description = FindPackageShare("so101_description").find("so101_description")
    _extend_gz_resource_path(os.path.join(pkg_description, ".."))
    
    #     - Carga del archivo Xacro y descripción del robot
    xacro_file = os.path.join(pkg_description, "urdf", "so101_new_calib.urdf.xacro")
    robot_description = ParameterValue(Command(["xacro ", xacro_file, " mode:=gazebo"]), value_type=str)

    #     - Inicialización del nodo publisher de estado
    #           use_sim_time=True es obligatorio para que los timestamps de TF2 coincidan
    #           con el reloj de Gazebo. Sin esto, lookup_transform en PoseTracker falla
    #           porque /tf publica con tiempo de simulación pero el nodo usa tiempo real.
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
    )

    # Inicialización del directorio del mundo de Gazebo
    pkg_gazebo_share = get_package_share_directory("so101_gazebo")
    world_path = os.path.join(pkg_gazebo_share, "worlds", "complete_main.sdf")

    #     - Inicialización Launcher del mundo de gazebo
    #     - Se le pasan argumentos para que arranque en modo repetición (-r) y con el mundo cargado (-s)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare("ros_gz_sim").find("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={"gz_args": f"-r -s {world_path}"}.items(),
    )

    # Spawner del robot en Gazebo
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "so101_new_calib",
            "-x", "0.0", "-y", "0.0", "-z", "0.0",
        ],
    )

    # --- PUENTE ROS-GAZEBO ---
    # Argumentos:
    #   - /clock: para sincronizar el tiempo entre ROS y Gazebo (esencial para controladores y TF2)
    #   - /camera/image_raw: imagen de la cámara RGB del robot
    #   - /camera_caballera/image_raw: imagen de la cámara de cab
    #   - /camera/camera_info: información de la cámara (resolución, intrínsecos)
    #   - /world/main1_world/dynamic_pose/info: para obtener la posición del cubo desde Gazebo
    #   - /world/main1_world/contact: para obtener información de contactos (colisiones) desde Gazebo, útil para detectar agarres exitosos
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera_caballera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/world/main1_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            "/world/main1_world/contact@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts",
        ],
    )

    # --- CONTROLADORES DEL ROBOT ---
    #     - Nodo de estado del robot
    load_jsb = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager-timeout", "180"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    #   - Controlador de los joints del brazo
    load_arm = Node(
        package="controller_manager", executable="spawner",
        arguments=["arm_controller", "--controller-manager-timeout", "60"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    #   - Controlador de la pinza
    load_gripper = Node(
        package="controller_manager", executable="spawner",
        arguments=["gripper_controller", "--controller-manager-timeout", "60"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # Manejo de eventos para arrancar controladores en orden
    start_jsb_after_spawn = RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=[load_jsb]))
    start_arm_after_jsb = RegisterEventHandler(OnProcessExit(target_action=load_jsb, on_exit=[load_arm]))
    start_gripper_after_arm = RegisterEventHandler(OnProcessExit(target_action=load_arm, on_exit=[load_gripper]))

    # Visor de Cámara (desactivar con use_rqt:=false durante entrenamiento para ahorrar recursos)
    use_rqt = LaunchConfiguration("use_rqt")
    declare_use_rqt = DeclareLaunchArgument(
        "use_rqt",
        default_value="false",  # FIX 5: false por defecto — durante entrenamiento no se necesita
        description="Launch rqt_image_view (desactivar en entrenamiento: use_rqt:=false)"
    )

    rqt_image_view = Node(
        package="rqt_image_view", executable="rqt_image_view", name="rqt_image_view",
        output="screen", arguments=["/camera/image_raw"], condition=IfCondition(use_rqt),
    )

    return LaunchDescription(
        [
            declare_use_rqt,
            robot_state_publisher,
            gazebo,
            spawn_robot,
            gz_bridge,
            start_jsb_after_spawn,
            start_arm_after_jsb,
            start_gripper_after_arm,
            rqt_image_view,
        ]
    )