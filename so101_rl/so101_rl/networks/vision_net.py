import torch
import torch.nn as nn
import numpy as np

class VisionFusionNet(nn.Module):
    def __init__(self, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        
        # Se define una red convolucional (CNN)
        #   - imagen 84x84
        #   - Blanco y Negro (1 canal)
        # Salida: vector de 2592 características
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), 
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), 
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # Red Lineal (MLP) propioceptiva 
        #   - (9 valores: 6 joints + 3 coordenadas objetivo)
        #   - Salida de 64 neuronas
        self.mlp_state = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(inplace=True)
        )

        # 3. Fusión
        action_dim = np.prod(action_shape) if action_shape else 0
        fusion_dim = 2592 + 64 + action_dim  # CNN_out + MLP_out + Actions (si es Crítico)
        
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        self.output_dim = 128

    def forward(self, obs, state=None, info={}):
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        # TRUCO DE TIELSHOU: Separamos el vector gigante de vuelta en Imagen y Propiocepción
        # Los primeros 7056 elementos son la imagen (84 * 84 = 7056)
        img_flat = obs_tensor[:, :7056]
        # Los siguientes 9 elementos son los joints y posición del cubo
        prop_state = obs_tensor[:, 7056:7065]
        
        # Si es el crítico de SAC, las acciones vienen pegadas al final del vector obs
        action_tensor = None
        if obs_tensor.shape[1] > 7065:
            action_tensor = obs_tensor[:, 7065:]

        # Reconstruimos la imagen plana a una matriz 2D (Batch, Channels, Height, Width)
        img_cnn = img_flat.view(-1, 1, 84, 84) / 255.0  

        # Pasamos los datos por las dos redes en paralelo
        cnn_out = self.cnn(img_cnn)
        state_out = self.mlp_state(prop_state)

        # Fusionamos las "ideas" de la cámara y de los sensores
        if action_tensor is not None:
            fused = torch.cat([cnn_out, state_out, action_tensor], dim=1)
        else:
            fused = torch.cat([cnn_out, state_out], dim=1)

        # Decisión final
        logits = self.fc(fused)
        return logits, state