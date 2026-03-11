# Entrenamiento HER + SAC para el entorno SO101
# Autor: Víctor Gil
# Contacto: 
#   - Correo Institucional: victor.gil107@alu.ulpgc.es
#   - Correo Personal: gbvictor011@protonmail.com
#   - GitHub: BeepBoopVictor
# Este código implementa un agente de aprendizaje por refuerzo utilizando la técnica de Hindsight Experience Replay (HER) combinada con el algoritmo Soft Actor-Critic (SAC) para el entorno SO101.
# Este entrenamiento se centra en enseñar al agente a acercarse al cubo (Reaching)

import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, HERVectorReplayBuffer  # 
from tianshou.env import DummyVectorEnv
from tianshou.policy import SACPolicy   # https://tianshou.org/en/v1.1.0/03_api/policy/modelfree/sac.html#tianshou.policy.modelfree.sac.SACPolicy 07/03/26
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic # https://tianshou.org/en/v1.1.0/03_api/utils/net/continuous.html 07/03/26

from so101_rl.envs.so101_env_her import SO101HEREnv
from so101_rl.utils.metrics_logger import MetricsLogger



# Función de recompensa Sparse
# def compute_reward(achieved_goal, desired_goal, info=None):
#     """
#     Función de recompensa dispersa para HER
    
#     Recompensa: 0 si el agente está a menos de 5cm del objetivo, -1 en caso contrario.
#     """

#     # Distancia euclediana entre el objetivo alcanzado y el deseado.
#     d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    
#     return -(d > 0.05).astype(np.float32)

# Función de recompensa Densa + Sparse
def compute_reward(achieved_goal, desired_goal, info=None):
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    # sparse = -(d > 0.05).astype(np.float32)
    sparse = -(d > 0.15).astype(np.float32)
    dense  = -d * 0.1
    return sparse + dense


class HERCritic(torch.nn.Module):
    """Evalua Q(s,a) dado el estado (observación + meta) y la acción."""

    def __init__(self, action_shape, device='cpu'):
        super().__init__()
        self.device = device
        action_dim = np.prod(action_shape)
        
        # Red neuronal simple para el crítico.
        # Recibe:
        #   - 15 (estado: 6Dof + 3 posicion pinza + 3 posicion del cubo) + 3 (meta: x,y,z cubo rojo) + 6 (acciones) = 21 entradas
        # Salida:
        #   - 1 valor Q para la acción dada el estado y la meta
        # NOTA: el hecho de que se repita la posición del cubo ambas veces  es debido al HER,
        # ya que cumple tanto el rol de estado como de meta.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(15 + action_dim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1)
        )

    def forward(self, obs, act=None, info={}):
        """El crítico recibe observación (estado + meta) y acción, y devuelve el valor Q correspondiente."""

        if hasattr(obs, 'observation') and hasattr(obs, 'desired_goal'):
            o = torch.as_tensor(obs.observation, device=self.device, dtype=torch.float32)   # Estado actual del robot
            g = torch.as_tensor(obs.desired_goal, device=self.device, dtype=torch.float32)  # Meta (posición del cubo)
        else:
            # Si el formato de obs no es el esperado, lanzamos un error claro para facilitar la depuración.
            raise ValueError(f"El Crítico esperaba un dict/Batch con 'observation' y 'desired_goal', recibió: {type(obs)}")
        
        if o.dim() == 1:    # Si recibimos un solo estado (no batch), añadimos una dimensión de batch para que la red lo procese correctamente.
            o = o.unsqueeze(0)
            g = g.unsqueeze(0)
            
        a = torch.as_tensor(act, device=self.device, dtype=torch.float32) # Acción tomada por el agente
        if a.dim() == 1:
            a = a.unsqueeze(0)
            
        x = torch.cat([o, g, a], dim=1) # Concatenamos estado, meta y acción para alimentar a la red del crítico
        
        return self.net(x)  # Devolvemos el valor Q estimado para esa combinación de estado, meta y acción.


class HERFeatureNet(torch.nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        # Red neuronal simple para el crítico.
        # Recibe:
        #   - 12 (estado: 6Dof + 3 posicion pinza + 3 posicion del cubo) + 3 (meta: x,y,z cubo rojo) = 15 entradas
        # Salida:
        #   - 256 características para alimentar al Actor y a los Críticos.
        # NOTA: se excluyen las 6 entradas de acción porque esta red se usará para el Actor, que no recibe la acción como entrada.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(15, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True)
        )
        self.output_dim = 256
        
    def forward(self, obs, state=None, info={}):
        """El Actor recibe observación (estado + meta) y devuelve características para la política."""

        if hasattr(obs, 'observation') and hasattr(obs, 'desired_goal'):
            o = torch.as_tensor(obs.observation, device=self.device, dtype=torch.float32)   # Estado actual del robot
            g = torch.as_tensor(obs.desired_goal, device=self.device, dtype=torch.float32)  # Meta (posición del cubo)
        else:
            raise ValueError(f"El Actor esperaba un dict/Batch con 'observation' y 'desired_goal', recibió: {type(obs)}")
        
        if o.dim() == 1:
            o = o.unsqueeze(0)
            g = g.unsqueeze(0)
            
        x = torch.cat([o, g], dim=1)    # Concatenación de estado y meta para alimentar a la red del actor
        
        return self.net(x), state   # Devolvemos las características extraídas y el estado oculto


def main():
    print("Inicializando Entorno HER...")
    env = SO101HEREnv()
    
    obs_shape = env.observation_space['observation'].shape
    action_shape = env.action_space.shape
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo de entrenamiento: {device.upper()}")

    # Logger
    log_path = os.path.join("log", "sac_her_so101")
    metrics = MetricsLogger(log_dir=log_path)

    # Vector de entornos: Tianshou entrena múltiples entornos en paralelo. En este caso, solo se usa uno.
    train_envs = DummyVectorEnv([lambda m=metrics: SO101HEREnv(logger=m)])
    test_envs = DummyVectorEnv([lambda: SO101HEREnv()])

    # --- REDES NEURONALES (MLP) ---
    # ActorProb: no es determinista, devuelve una distribución de probabilidad sobre las acciones.
    #   - Inicialización de la red del actor
    net_a = HERFeatureNet(device=device).to(device)
    #   - El Actor recibe los 256 features de HERFeatureNet y devuelve una distribución (mu, sigma)
    actor = ActorProb(net_a, action_shape, max_action=env.action_space.high[0], device=device, unbounded=True).to(device)
    
    # Críticos: reciben el estado, la meta y la acción, y devuelven el valor Q correspondiente.
    #  - Clipped Double Q-Learning: se usan dos críticos para reducir el sesgo positivo en la estimación de Q. 
    #    Ambos críticos tienen la misma arquitectura pero parámetros independientes.
    #    Se cree al crítico más pesimista, para evitar sobreestimar el valor de las acciones: Q(s,a) = min(Q1(s,a), Q2(s,a))
    critic1 = HERCritic(action_shape, device=device).to(device)
    critic2 = HERCritic(action_shape, device=device).to(device)

    # --- POLÍTICA SAC ---
    # SAC: algoritmo de aprendizaje por refuerzo off-policy que optimiza una política estocástica maximizando la entropía de la política además de la recompensa esperada.
    #    - Se introducen los actores y críticos.
    #    - Se optimizan con Adam a una tasa de aprendizaje de 1e-3.
    #    - Se especifica el espacio de acción para que la política pueda normalizar las acciones correctamente.
    policy = SACPolicy(
        actor,   torch.optim.Adam(actor.parameters(),   lr=3e-4),
        critic1, torch.optim.Adam(critic1.parameters(), lr=3e-4),
        critic2, torch.optim.Adam(critic2.parameters(), lr=3e-4),
        action_space=env.action_space
    )

    # --- HER REPLAY BUFFER ---
    # https://tianshou.org/en/latest/_modules/tianshou/data/buffer/her.html 05/03/26
    # HERVectorReplayBuffer: buffer de experiencia que implementa Hindsight Experience Replay (HER) para entornos con objetivos.
    #    - Size: capacidad máxima de 100.000 transiciones.
    #    - Num_envs: número de entornos paralelos (en este caso, 1).
    #    - compute_reward_fn: función de recompensa personalizada (distancia entre la posición alcanzada y la deseada).
    #    - Horizon: longitud máxima de un episodio (150 pasos).
    #    - Future_k: proporción de transiciones que se reetiquetarán. 4 reetiquetados por cada transición original.
    buffer = HERVectorReplayBuffer(
        100000,                            
        len(train_envs),                   
        compute_reward_fn=compute_reward,
        horizon=150,                       
        future_k=4.0                      
    )

    # --- COLECTORES ---
    # Collector: se encarga de interactuar con el entorno y almacenar las transiciones en el buffer.
    #   - train_collector: se usa para recolectar experiencias durante el entrenamiento. Se activa el ruido de exploración para fomentar la exploración del espacio de acciones.
    #   - test_collector: evalua la política sin ruido de exploración, para obtener una medida más precisa del rendimiento del agente.
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # --- LOGGER Y FUNCION DE GUARDADO ---
    # Logger: se utiliza TensorBoard para visualizar métricas de entrenamiento como la recompensa promedio, la longitud de los episodios, etc.
    #     - save_best_fn: función que se llama cada vez que se obtiene un nuevo mejor resultado durante las pruebas. 
    #       Guarda los pesos de la política en un archivo 'policy_her.pth' dentro del directorio de logs.
    os.makedirs(log_path, exist_ok=True)
    logger = TensorboardLogger(SummaryWriter(log_path))
    save_best_fn = lambda policy: torch.save(policy.state_dict(), os.path.join(log_path, 'policy_her.pth'))

    # --- RECOLECCIÓN DE DATOS INICIAL ---
    # Antes de comenzar el entrenamiento, se llena el buffer de experiencia con algunas transiciones aleatorias para que el agente tenga datos con los que aprender desde el principio.
    print("Recolectando datos iniciales (llenando el Replay Buffer)...")
    train_collector.collect(n_step=1000, random=True)

    print("Iniciando entrenamiento principal (Fase 1: Estático)...")

    def on_epoch_end(epoch, env_step):
        summary = metrics.log_epoch_end()
        metrics.new_epoch()           
        print(f"[Época {epoch}] éxito={summary.get('success_rate',0):.1%} | "
                f"dist_min={summary.get('mean_min_distance',0):.3f}m | "
                f"reward={summary.get('mean_reward',0):.3f}")

    # --- BUCLE DE ENTRENAMIENTO ---
    # Se entrena durante un máximo de 30 épocas, con 2000 pasos por época.
    #   - policy: Implementación de SAC que define cómo se actualizan el Actor y los Críticos.
    #   - train_collector / test_collector: Encargados de ejecutar la política en el entorno para obtener datos o evaluar.
    #   - max_epoch: Duración total del experimento (30 épocas). Una época es un bloque de pasos antes de una evaluación.
    #   - step_per_epoch: Define la frecuencia de diagnóstico. Cada 2000 pasos se pausa el entrenamiento para testear y loguear.
    #   - step_per_collect: Granularidad de la interacción. El agente interactúa 10 pasos con el entorno antes de intentar actualizar las redes.
    #   - update_per_step: Gradiente de actualización (0.1). Determina que se realizará 1 actualización de pesos por cada 10 pasos de simulación (Ratio 1:10).
    #   - episode_per_test: Cantidad de episodios realizados durante la fase de evaluación para obtener una métrica de recompensa estable y sin ruido.
    #   - batch_size: Cantidad de transiciones (estado, acción, meta, recompensa) extraídas del HER Buffer para cada paso de optimización.
    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=100,
        step_per_epoch=2000,
        # step_per_collect=10,
        step_per_collect=20,
        update_per_step=0.1,
        episode_per_test=10,
        batch_size=256,
        save_best_fn=save_best_fn,
        logger=logger,
        test_fn=on_epoch_end,
    )

    print(f"\n¡Entrenamiento Finalizado!\nResultados: {result}")
    env.close(); train_envs.close(); test_envs.close()

if __name__ == '__main__':
    main()