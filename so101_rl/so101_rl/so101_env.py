import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import time
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from cv_bridge import CvBridge

# Mensajes de ROS 2
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand  # Asumiendo que usa este estándar

class SO101Env(gym.Env):
    def __init__(self):
        super(SO101Env, self).__init__()
        
        # --- 1. CONFIGURACIÓN DE GIMNASIO (RL) ---
        # Acción: 5 joints del brazo + 1 pinza = 6 valores continuos entre -1 y 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Observación: Diccionario con Imagen y Joints
        # Reducimos la imagen de 649x480 a algo manejable para RL (ej. 84x84 en escala de grises)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
            "joints": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        })

        # --- 2. CONFIGURACIÓN DE ROS 2 ---
        rclpy.init()
        self.node = rclpy.create_node('tianshou_so101_env')
        self.bridge = CvBridge()

        # Nombres de las articulaciones del brazo
        self.arm_joint_names = [
            'shoulder_pan', 'shoulder_lift', 'elbow_flex', 
            'wrist_flex', 'wrist_roll'
        ]

        # Publicador del Brazo
        self.arm_pub = self.node.create_publisher(
            JointTrajectory, 
            '/arm_controller/joint_trajectory', 
            10
        )

        # Cliente de Acción de la Pinza
        self.gripper_client = ActionClient(
            self.node, 
            GripperCommand, 
            '/gripper_controller/gripper_cmd'
        )

        # Suscriptores
        self.image_sub = self.node.create_subscription(
            Image, '/camera/image_raw', self._image_callback, 10)
        self.joint_sub = self.node.create_subscription(
            JointState, '/joint_states', self._joint_callback, 10)

        # Variables de estado
        self.latest_image = np.zeros((84, 84, 1), dtype=np.uint8)
        self.latest_joints = np.zeros(6, dtype=np.float32)

        # --- 3. HILO EN SEGUNDO PLANO PARA ROS 2 ---
        # Esto permite que ROS 2 reciba mensajes sin bloquear el entrenamiento de Tianshou
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        # Esperar a que el Action Server de la pinza esté listo
        self.gripper_client.wait_for_server(timeout_sec=5.0)

    def _image_callback(self, msg):
        """Procesa la imagen a 84x84 en escala de grises para la CNN."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        resized = cv2.resize(cv_image, (84, 84))
        self.latest_image = np.expand_dims(resized, axis=-1)

    def _joint_callback(self, msg):
        """Actualiza el estado de los joints."""
        # Nota: En un caso real, debes mapear los nombres de msg.name con sus posiciones
        # para asegurarte de que el orden sea siempre el mismo.
        if len(msg.position) >= 6:
            self.latest_joints = np.array(msg.position[:6], dtype=np.float32)

    def step(self, action):
        # 1. Separar acciones (desnormalizar de [-1, 1] a radianes si es necesario)
        arm_action = action[:5]
        gripper_action = action[5]

        # 2. Enviar comando al Brazo
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = arm_action.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000 # 0.1 segundos
        traj_msg.points = [point]
        self.arm_pub.publish(traj_msg)

        # 3. Enviar comando a la Pinza (Asíncrono para no bloquear)
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(gripper_action)
        self.gripper_client.send_goal_async(goal_msg)

        # 4. Dar tiempo a la simulación para que avance (Sustituir por reloj de simulación más adelante)
        time.sleep(0.1)

        # 5. Recoger observaciones
        obs = {
            "image": self.latest_image.copy(),
            "joints": self.latest_joints.copy()
        }

        # 6. Lógica de Recompensa y Fin de Episodio (¡A programar más adelante!)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Definir la posición "Home" del brazo y la pinza
        # (Si todos ceros hace que el robot choque contra el suelo, cámbialo por una pose segura)
        home_arm_positions = [0.0, 0.0, 0.0, 0.0, 0.0] 
        home_gripper_position = 0.0  # Asumimos que 0.0 es pinza abierta
        
        # 2. Enviar comando "Home" al brazo
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = home_arm_positions
        # Le damos 1 segundo para hacer el movimiento suavemente
        point.time_from_start.sec = 1 
        point.time_from_start.nanosec = 0
        traj_msg.points = [point]
        self.arm_pub.publish(traj_msg)

        # 3. Enviar comando para abrir la pinza
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = home_gripper_position
        self.gripper_client.send_goal_async(goal_msg)

        # 4. Esperar a que el robot llegue a la posición Home
        tolerance = 0.05  # Margen de error en radianes aceptable
        max_wait_time = 2.0  # Segundos máximos a esperar para no colgar el entrenamiento
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Comparamos la posición actual de los 5 joints del brazo con la posición Home
            current_arm_joints = self.latest_joints[:5]
            error = np.max(np.abs(current_arm_joints - np.array(home_arm_positions)))
            
            if error < tolerance:
                break # ¡El robot ha llegado a su posición inicial!
            time.sleep(0.05) # Pausa pequeña para no saturar la CPU

        # TODO futuro: Aquí meteremos el comando `ign service` vía subprocess 
        # para reaparecer el cubo/objeto en la mesa.
        
        # 5. Devolver la primera observación real del nuevo episodio
        obs = {
            "image": self.latest_image.copy(),
            "joints": self.latest_joints.copy()
        }
        return obs, {}
