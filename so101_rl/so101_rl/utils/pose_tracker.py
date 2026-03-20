# pose_tracker.py
# Proyecto: TFG - Implementación de modelos de inteligencia artificial en un entorno robótico simulado.
# Institución: Universidad de Las Palmas de Gran Canaria
# Autor: Víctor Gil Bernal
#
# ARQUITECTURA DE FUENTES:
#   - Posición del gripper (ee_frame): leída desde TF2 (/tf + /tf_static)
#     El robot_state_publisher publica las transformaciones de los links del robot en TF2.
#     ros2_control actualiza esas transformaciones en cada ciclo de control.
#
#   - Posición del cubo (target_frame): leída desde /world/main1_world/dynamic_pose/info
#     Ignition Fortress publica aquí las poses de los cuerpos físicos dinámicos (modelos SDF).
#     El robot NO aparece en este topic porque sus links son gestionados por ros2_control, no por
#     el motor de física de Ignition directamente.
#
# PROBLEMA DEL DISEÑO ANTERIOR:
#   Ambas poses se leían desde dynamic_pose/info. El gripper nunca aparecía en ese topic,
#   por lo que ee_pos se quedaba congelada en su valor inicial (0,0,0), haciendo que la
#   distancia observada fuera constante durante largos periodos de entrenamiento.

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener


class PoseTracker:
    def __init__(self, node, ee_frame='gripper_frame_link'):
        self.node = node
        self.ee_frame = ee_frame

        # Posiciones de ambos cubos
        self.red_cube_pos  = np.zeros(3, dtype=np.float32)
        self.blue_cube_pos = np.zeros(3, dtype=np.float32)
        self.ee_pos        = np.zeros(3, dtype=np.float32)

        self._red_received  = False
        self._blue_received = False
        self._ee_received   = False

        self.sub = self.node.create_subscription(
            TFMessage,
            '/world/main1_world/dynamic_pose/info',
            self._dynamic_pose_callback,
            10
        )

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.tf_timer    = self.node.create_timer(0.02, self._update_ee_from_tf)
        self.reference_frame = 'world'

    def _dynamic_pose_callback(self, msg):
        for transform in msg.transforms:
            frame = transform.child_frame_id
            t = transform.transform.translation
            if 'red_cube' in frame:
                self.red_cube_pos  = np.array([t.x, t.y, t.z], dtype=np.float32)
                self._red_received = True
            elif 'blue_cube' in frame:
                self.blue_cube_pos  = np.array([t.x, t.y, t.z], dtype=np.float32)
                self._blue_received = True

    def _update_ee_from_tf(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.reference_frame, self.ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            t = tf.transform.translation
            self.ee_pos = np.array([t.x, t.y, t.z], dtype=np.float32)
            self._ee_received = True
        except Exception:
            pass

    def get_active_cube_pos(self, color: str) -> np.ndarray:
        """Devuelve la posición del cubo activo según el color."""
        return self.red_cube_pos if color == 'red' else self.blue_cube_pos

    def is_ready(self) -> bool:
        return self._red_received and self._blue_received and self._ee_received