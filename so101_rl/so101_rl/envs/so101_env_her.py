# so101_env_her.py - Entorno de Aprendizaje por Refuerzo con HER para el SO101
# Proyecto: TFG - Implementación de modelos de inteligencia artificial en un entorno robótico simulado.
# Institución: Universidad de Las Palmas de Gran Canaria
# Autor: Víctor Gil Bernal
# Contacto: 
#   - Correo Institucional: victor.gil107@alu.ulpgc.es
#   - Correo Personal: gbvictor011@protonmail.com
#   - GitHub: BeepBoopVictor
# Última actualización: 09/03/2026
# Este código define un entorno de aprendizaje por refuerzo personalizado para el robot SO101 utilizando Hindsight Experience Replay (HER).

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import time
import subprocess
import random
import math
import uuid

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState

from so101_rl.utils.pose_tracker import PoseTracker

CUBE_ORIGINS = {
    'red':  np.array([-0.1, -0.2, 0.02], dtype=np.float32),
    'blue': np.array([-0.1,  0.2, 0.02], dtype=np.float32),
}

class SO101HEREnv(gym.Env):
    """Entorno de Aprendizaje por Refuerzo con HER para el SO101"""

    def __init__(self, conveyor_speed=0.0, logger=None):
        super(SO101HEREnv, self).__init__()
        
        if not rclpy.ok(): rclpy.init()
            
        # Nodo único del entorno con un identificador aleatorio para evitar conflictos en ROS 2
        node_name = f'tianshou_so101_her_env_{uuid.uuid4().hex[:6]}'
        self.node = rclpy.create_node(node_name)
        
        # Parámetro futuro para implementar velocidad variable de la cinta
        self.conveyor_speed = conveyor_speed

        # Logger de métricas
        self.metrics = logger
        
        # --- ESPACIO DE OBSERVACIÓN ---
        # Acción: 6 joints continuos [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)


        self.tracker = PoseTracker(self.node, ee_frame='gripper_frame_link')

        # Color del cubo activo en este episodio
        self.active_color = 'red'  # se sortea en reset()

        # 'observation': [6 joints, obj_x, obj_y, obj_z] (9 valores)
        # 'achieved_goal': Posición actual de la pinza (X,Y,Z)
        # 'desired_goal': Posición actual del cubo (X,Y,Z)
        # Quita el target_frame del tracker, ahora trackea los dos
        # observation: añade 1 valor para el color (one-hot: 0=rojo, 1=azul)
        self.observation_space = spaces.Dict({
            # 'observation': spaces.Box(low=-10.0, high=10.0, shape=(13,), dtype=np.float32),
            'observation': spaces.Box(low=-10.0, high=10.0, shape=(10,), dtype=np.float32),
            # [6 joints | 3 gripper_pos | 3 cube_pos | 1 color]
            'achieved_goal': spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
            'desired_goal':  spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
        })

        # --- ROS 2 SETUP ---
        self.arm_joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
        self.arm_pub = self.node.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_client = ActionClient(self.node, GripperCommand, '/gripper_controller/gripper_cmd')
        
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_callback, 10)
        self.latest_joints = np.zeros(6, dtype=np.float32)

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        self.gripper_client.wait_for_server(timeout_sec=5.0)
        
        self.current_step = 0
        self.max_steps = 200 
        
        # Definimos el umbral de éxito (10 cm)
        self.distance_threshold = 0.10

        self.JOINT_LIMITS = np.array([
            [-1.91986, 1.91986],  # shoulder_pan
            [-1.74533, 1.74533],  # shoulder_lift
            [-1.69,    1.69   ],  # elbow_flex
            [-1.65806, 1.65806],  # wrist_flex
            [-2.74385, 2.84121],  # wrist_roll
        ], dtype=np.float32)

    def _joint_callback(self, msg):
        if len(msg.position) >= 6:
            self.latest_joints = np.array(msg.position[:6], dtype=np.float32)


    # Función para cambiar la velocidad de la cinta en caliente (Curriculum)
    def set_conveyor_speed(self, speed):
        self.conveyor_speed = speed
    

    def _get_obs(self):
        cube_pos    = self.tracker.get_active_cube_pos(self.active_color).astype(np.float32)
        gripper_pos = self.tracker.ee_pos.astype(np.float32)
        color_flag  = np.array([0.0 if self.active_color == 'red' else 1.0], dtype=np.float32)

        obs_array = np.concatenate([
            self.latest_joints[:6],  # 6
            gripper_pos,             # 3
            # cube_pos,                # 3
            color_flag               # 1
        ]).astype(np.float32)        # total: 13

        return {
            'observation':    obs_array,
            'achieved_goal':  gripper_pos,
            'desired_goal':   cube_pos,
        }


    # --- COMPUTE REWARD ---
    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        sparse = -(d > self.distance_threshold).astype(np.float32)
        d_clipped = np.clip(d, 0.0, 1.0)
        dense  = -d_clipped * 0.4
        
        return sparse + dense

    def step(self, action):
        self.current_step += 1
        
        # Ejecutar acción
        raw_arm_action = action[:5]
        gripper_action = action[5]

        # 1. Aplicar inercia (Low-Pass Filter)
        momentum = 0.4
        smoothed_action = (momentum * self.previous_arm_action) + ((1.0 - momentum) * raw_arm_action)
        self.previous_arm_action = smoothed_action 

        # 2. Control Tracking puro (Ignoramos los sensores para la orden de movimiento)
        max_delta = 0.10  
        self.target_joint_positions += (smoothed_action * max_delta)
        # self.target_joint_positions = np.clip(self.target_joint_positions, -3.0, 3.0)

        self.target_joint_positions = np.clip(
            self.target_joint_positions,
            self.JOINT_LIMITS[:, 0],
            self.JOINT_LIMITS[:, 1]
        )
        
        # 3. Empaquetar el mensaje ROS
        traj_msg = JointTrajectory(joint_names=self.arm_joint_names)
        point = JointTrajectoryPoint()
        
        # --- Usamos la variable suavizada correcta ---
        point.positions = self.target_joint_positions.tolist()
        
        # --- Igualamos el tiempo del motor a 0.2s ---
        point.time_from_start.nanosec = 200000000 # 0.2s (200 millones de nanosegundos)
        

        traj_msg.points = [point]
        self.arm_pub.publish(traj_msg)

        goal_msg = GripperCommand.Goal()

        # Mapeo lineal: acción [-1, 1]: posición real del gripper [lower, upper]
        # action=-1: gripper cerrado (-0.174533 rad)
        # action=+1: gripper abierto (+1.74533 rad)
        gripper_low, gripper_high = -0.174533, 1.74533
        gripper_pos = gripper_low + (float(gripper_action) + 1.0) / 2.0 * (gripper_high - gripper_low)
        goal_msg.command.position = float(np.clip(gripper_pos, gripper_low, gripper_high))
        goal_msg.command.max_effort = 10.0

        self.gripper_client.send_goal_async(goal_msg)

        # Sincronización con el reloj de Gazebo en vez de sleep real bloqueante.
        # Esperamos hasta que el topic /clock avance al menos 0.2s de tiempo de simulación.
        # Esto es más eficiente si la simulación corre más rápido que real-time.
        t_start = self.node.get_clock().now()
        while (self.node.get_clock().now() - t_start).nanoseconds < 200_000_000:
            time.sleep(0.005)

        # --- DICCIONARIOS Y RECOMPENSAS ---
        obs_dict = self._get_obs()
        
        reward = float(self.compute_reward(obs_dict['achieved_goal'], obs_dict['desired_goal'], {}))

        distance = float(np.linalg.norm(obs_dict['achieved_goal'] - obs_dict['desired_goal']))
        is_success = distance < self.distance_threshold

        # Terminación anticipada si el cubo cae del área de trabajo (z < 0.01m)
        cube_fell = obs_dict['desired_goal'][2] < 0.01

        terminated = is_success
        truncated = self.current_step >= self.max_steps or cube_fell

        info = {
            "distance_to_cube": distance,
            "is_success": 1.0 if is_success else 0.0,
            "cube_fell": 1.0 if cube_fell else 0.0,
        }

        if self.metrics:
            # print(f"[DEBUG] log_step llamado | dist={info['distance_to_cube']:.3f} | reward={reward:.3f} | success={is_success}")
            self.metrics.log_step(
                reward   = reward,
                distance = info['distance_to_cube'],
                gripper_pos = self.tracker.ee_pos,
                cube_pos    = self.tracker.get_active_cube_pos(self.active_color),
            )
            if terminated or truncated:
                print(f"[EP] dist={info['distance_to_cube']:.3f} | reward={reward:.3f} | success={is_success} | color={self.active_color}")
                self.metrics.log_episode_end(success=is_success)

        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.active_color  = random.choice(['red', 'blue'])
        inactive_color     = 'blue' if self.active_color == 'red' else 'red'

        home_arm_positions = [0.0, -0.4, 0.2, 0.5, 0.0]
        self.target_joint_positions = np.array(home_arm_positions, dtype=np.float32)
        self.previous_arm_action    = np.zeros(5, dtype=np.float32)

        # 1. Primero mover el robot a home
        traj_msg = JointTrajectory(joint_names=self.arm_joint_names)
        point = JointTrajectoryPoint()
        point.positions = home_arm_positions
        point.time_from_start.nanosec = 500_000_000
        traj_msg.points = [point]
        self.arm_pub.publish(traj_msg)

        gripper_goal = GripperCommand.Goal()
        gripper_goal.command.position  = 1.5
        gripper_goal.command.max_effort = 10.0
        self.gripper_client.send_goal_async(gripper_goal)

        # 2. Esperar a que el robot llegue a home ANTES de mover los cubos
        start_time = time.time()
        while time.time() - start_time < 2.0:
            if np.max(np.abs(self.latest_joints[:5] - np.array(home_arm_positions))) < 0.05:
                break
            time.sleep(0.05)

        # 3. Solo AHORA mover los cubos — el robot ya está quieto en home
        self._set_cube_pose(inactive_color, CUBE_ORIGINS[inactive_color])

        radio  = random.uniform(0.28, 0.42)
        angulo = random.uniform(-0.55, 0.55)
        candidate_pos = np.array([radio * math.cos(angulo), radio * math.sin(angulo), 0.04])

        attempts = 0
        while attempts < 10:
            candidate_pos = np.array([
                radio * math.cos(angulo),
                radio * math.sin(angulo),
                0.04
            ])
            if np.linalg.norm(candidate_pos - self.tracker.ee_pos) > self.distance_threshold + 0.08:
                break
            radio  = random.uniform(0.28, 0.42)
            angulo = random.uniform(-0.55, 0.55)
            attempts += 1

        self._set_cube_pose(self.active_color, candidate_pos)

        # Dar tiempo a dynamic_pose/info para publicar la nueva posición
        time.sleep(0.15)

        # 4. Esperar tracker
        timeout = time.time() + 3.0
        while not self.tracker.is_ready() and time.time() < timeout:
            time.sleep(0.05)

        return self._get_obs(), {}

    def close(self):
        try:
            if hasattr(self, 'node') and self.node is not None:
                self.executor.remove_node(self.node)
                self.node.destroy_node()
                self.node = None
                
            self.executor.shutdown()
            if hasattr(self, 'spin_thread') and self.spin_thread.is_alive():
                self.spin_thread.join(timeout=1.0)
        except Exception: pass


    def _set_cube_pose(self, color: str, position: np.ndarray):
        """Mueve un cubo a la posición indicada vía ign service."""
        name = f'{color}_cube'
        x, y, z = position
        req_str = (f'name: "{name}", '
                f'position: {{x: {x}, y: {y}, z: {z}}}, '
                f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}')
        cmd = [
            'ign', 'service', '-s', '/world/main1_world/set_pose',
            '--reqtype', 'ignition.msgs.Pose',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '2000', '--req', req_str
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(0.05)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception:
            pass