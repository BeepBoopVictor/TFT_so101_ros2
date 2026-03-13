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
        
        # 'observation': [6 joints, obj_x, obj_y, obj_z] (9 valores)
        # 'achieved_goal': Posición actual de la pinza (X,Y,Z)
        # 'desired_goal': Posición actual del cubo (X,Y,Z)
        self.observation_space = spaces.Dict({
            # 'observation': spaces.Box(low=-10.0, high=10.0, shape=(9,), dtype=np.float32), 
            'observation': spaces.Box(low=-10.0, high=10.0, shape=(12,), dtype=np.float32), 
            'achieved_goal': spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32), 
            'desired_goal': spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)   
        })

        # --- ROS 2 SETUP ---
        self.arm_joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
        self.arm_pub = self.node.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_client = ActionClient(self.node, GripperCommand, '/gripper_controller/gripper_cmd')
        
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_callback, 10)
        self.latest_joints = np.zeros(6, dtype=np.float32)

        self.tracker = PoseTracker(self.node, target_frame='red_cube', ee_frame='gripper_frame_link')

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        self.gripper_client.wait_for_server(timeout_sec=5.0)
        
        self.current_step = 0
        self.max_steps = 150 
        
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
        
        try:
            tf_time = self.tracker.last_tf_time  # necesitas exponer esto en PoseTracker
            ros_now = self.node.get_clock().now()
            age_ms = (ros_now - tf_time).nanoseconds / 1e6
            if age_ms > 150:
                print(f"[WARN] TF obsoleta: {age_ms:.0f}ms de antigüedad")
        except: pass

        cube_pos    = self.tracker.target_pos.astype(np.float32)
        gripper_pos = self.tracker.ee_pos.astype(np.float32)

        achieved_goal = gripper_pos
        desired_goal  = cube_pos

        obs_array = np.concatenate([
            self.latest_joints[:6],   # 6 joints
            gripper_pos,              # 3 valores: posición pinza
            cube_pos                  # 3 valores: posición cubo
        ]).astype(np.float32)         # total: 12 valores



        return {
            'observation': obs_array,
            'achieved_goal': achieved_goal,
            'desired_goal':  desired_goal
        }

    # --- COMPUTE REWARD ---
    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        sparse = -(d > self.distance_threshold).astype(np.float32)
        dense  = -d * 0.1
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
                cube_pos    = self.tracker.target_pos,
            )
            if terminated or truncated:
                print(f"[DEBUG] log_step llamado | dist={info['distance_to_cube']:.3f} | reward={reward:.3f} | success={is_success}")
                self.metrics.log_episode_end(success=is_success)

        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        home_arm_positions = [0.0, -0.4, 0.2, 0.5, 0.0]
        
        self.target_joint_positions = np.array(home_arm_positions, dtype=np.float32)
        
        self.previous_arm_action = np.zeros(5, dtype=np.float32)
        
        traj_msg = JointTrajectory(joint_names=self.arm_joint_names)
        point = JointTrajectoryPoint()
        point.positions = home_arm_positions
        point.time_from_start.nanosec = 500000000  # 0.5s para volver a casa suavemente
        traj_msg.points = [point]
        self.arm_pub.publish(traj_msg)


        gripper_goal = GripperCommand.Goal()
        gripper_goal.command.position = 1.5
        gripper_goal.command.max_effort = 10.0
        self.gripper_client.send_goal_async(gripper_goal)


        radio = random.uniform(0.22, 0.42)
        angulo = random.uniform(-0.55, 0.55) 
        cube_x, cube_y, cube_z = radio * math.cos(angulo), radio * math.sin(angulo), 0.04

        req_str = f'name: "red_cube", position: {{x: {cube_x}, y: {cube_y}, z: {cube_z}}}, orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
        cmd = ['ign', 'service', '-s', '/world/main1_world/set_pose', '--reqtype', 'ignition.msgs.Pose', '--reptype', 'ignition.msgs.Boolean', '--timeout', '2000', '--req', req_str]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(0.1)
            # Segundo set_pose para cancelar velocidad residual del cuerpo rígido
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except: pass

        start_time = time.time()
        while time.time() - start_time < 2.0:
            if np.max(np.abs(self.latest_joints[:5] - np.array(home_arm_positions))) < 0.05: break 
            time.sleep(0.05)
        
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