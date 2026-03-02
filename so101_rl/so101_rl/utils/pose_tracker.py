import numpy as np
from tf2_msgs.msg import TFMessage

class PoseTracker:
    def __init__(self, node, target_frame='red_cube', ee_frame='gripper_link'):
        self.node = node
        self.target_frame = target_frame
        # NOTA: Cambia 'gripper_link' por el nombre real del link final de tu robot 
        self.ee_frame = ee_frame 
        
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.ee_pos = np.zeros(3, dtype=np.float32)
        
        self.sub = self.node.create_subscription(
            TFMessage,
            '/world/main1_world/dynamic_pose/info',
            self._tf_callback,
            10
        )

    def _tf_callback(self, msg):
        for transform in msg.transforms:
            # Ignition Gazebo suele publicar el child_frame_id con el nombre del modelo o link
            frame_id = transform.child_frame_id
            
            if self.target_frame in frame_id:
                trans = transform.transform.translation
                self.target_pos = np.array([trans.x, trans.y, trans.z], dtype=np.float32)
                
            elif self.ee_frame in frame_id:
                trans = transform.transform.translation
                self.ee_pos = np.array([trans.x, trans.y, trans.z], dtype=np.float32)

    def get_distance(self):
        """Devuelve la distancia euclidiana entre la pinza y el objetivo."""
        return np.linalg.norm(self.ee_pos - self.target_pos)
