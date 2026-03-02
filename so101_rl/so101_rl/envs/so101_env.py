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

# Importamos nuestro nuevo tracker
from so101_rl.utils.pose_tracker import PoseTracker

class SO101Env(gym.Env):
    def __init__(self):
        super(SO101Env, self).__init__()
        
        if not rclpy.ok():
            rclpy.init()
            
        node_name = f'tianshou_so101_env_{uuid.uuid4().hex[:6]}'
        self.node = rclpy.create_node(node_name)
        
        # --- 1. CONFIGURACIÓN FASE 1 (Propiocepción) ---
        # Acción: 6 joints continuos [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Observación: 6 Joints + 3 Coordenadas (X,Y,Z) del cubo = 9 valores
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(9,), dtype=np.float32)

        # --- 2. ROS 2 SETUP ---
        self.arm_joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
        self.arm_pub = self.node.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_client = ActionClient(self.node, GripperCommand, '/gripper_controller/gripper_cmd')
        
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_callback, 10)
        self.latest_joints = np.zeros(6, dtype=np.float32)

        # Inicializamos el tracker de poses para las recompensas y observaciones
        self.tracker = PoseTracker(self.node, target_frame='red_cube', ee_frame='gripper_link') # Ajusta 'gripper_link' si es necesario

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        self.gripper_client.wait_for_server(timeout_sec=5.0)
        
        self.current_step = 0
        self.max_steps = 150 # Límite de tiempo por episodio

    def _joint_callback(self, msg):
        if len(msg.position) >= 6:
            self.latest_joints = np.array(msg.position[:6], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # 1. Ejecutar acción
        arm_action = action[:5]
        gripper_action = action[5]

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = arm_action.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000 # 0.1s
        traj_msg.points = [point]
        self.arm_pub.publish(traj_msg)

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(gripper_action)
        self.gripper_client.send_goal_async(goal_msg)

        time.sleep(0.1)

        # 2. Generar Observación (Joints + Posición del Cubo)
        obs = np.concatenate([self.latest_joints, self.tracker.target_pos]).astype(np.float32)

        # 3. Calcular Recompensa (Reward Shaping) y Finalización
        dist = self.tracker.get_distance()
        reward = -dist # Recompensa densa: penalizamos la distancia
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        is_success = False

        # Si está a menos de 4 cm, consideramos que ha "tocado" el cubo (Éxito Fase 1)
        if dist < 0.04:
            reward += 10.0
            terminated = True
            is_success = True

        # 4. Métricas para TensorBoard (Info dict)
        info = {
            "distance_to_cube": float(dist),
            "is_success": 1.0 if is_success else 0.0,
            "step_count": self.current_step
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 1. Reset Robot
        home_arm_positions = [0.0, -0.5, 0.5, 0.0, 0.0]
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = home_arm_positions
        point.time_from_start.sec = 1 
        traj_msg.points = [point]
        self.arm_pub.publish(traj_msg)

        # 2. Reset Cubo (Teletransporte Aleatorio)
        radio = random.uniform(0.12, 0.22)
        angulo = random.uniform(-0.6, 0.6) 
        cube_x, cube_y, cube_z = radio * math.cos(angulo), radio * math.sin(angulo), 0.015

        req_str = f'name: "red_cube", position: {{x: {cube_x}, y: {cube_y}, z: {cube_z}}}, orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
        cmd = ['ign', 'service', '-s', '/world/main1_world/set_pose', '--reqtype', 'ignition.msgs.Pose', '--reptype', 'ignition.msgs.Boolean', '--timeout', '2000', '--req', req_str]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except:
            pass

        # 3. Esperar estabilización
        start_time = time.time()
        while time.time() - start_time < 2.0:
            if np.max(np.abs(self.latest_joints[:5] - np.array(home_arm_positions))) < 0.05:
                break 
            time.sleep(0.05) 
            
        # Devolver observación inicial
        obs = np.concatenate([self.latest_joints, self.tracker.target_pos]).astype(np.float32)
        return obs, {}

    def close(self):
        try:
            # 1. Quitar el nodo del executor para que deje de procesar callbacks
            if hasattr(self, 'node') and self.node is not None:
                self.executor.remove_node(self.node)
                self.node.destroy_node()
                self.node = None
                
            # 2. Apagar el executor y el hilo de forma segura
            self.executor.shutdown()
            if hasattr(self, 'spin_thread') and self.spin_thread.is_alive():
                self.spin_thread.join(timeout=1.0)
        except Exception:
            pass
