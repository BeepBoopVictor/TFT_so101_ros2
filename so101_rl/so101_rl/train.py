import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

# Entorno propio de proyecto
from so101_rl.envs.so101_env import SO101Env

def main():

    # 1. Configuración del Entorno
    env = SO101Env()
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    
    # inicialización entornos vectorizados Tianshou
    train_envs = DummyVectorEnv([lambda: SO101Env()])
    test_envs = DummyVectorEnv([lambda: SO101Env()])

    # 2. Definición de las Redes (Actor y Critic para SAC)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    net_a = Net(state_shape, hidden_sizes=[128, 128], device=device)
    actor = ActorProb(net_a, action_shape, max_action=env.action_space.high[0], device=device, unbounded=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

    net_c1 = Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)

    net_c2 = Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    # 3. Política SAC
    policy = SACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        tau=0.005, gamma=0.99, alpha=0.2, estimation_step=1, action_space=env.action_space
    )

    # 4. Configurar el Logger de TensorBoard para las métricas
    log_path = os.path.join("log", "sac_so101_phase1")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    
    # 	- Se guarda el modelo:
    save_best_fn = lambda policy: torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # 5. Configurar Recolectores de datos
    buffer = VectorReplayBuffer(20000, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # Recolectar datos iniciales aleatorios para llenar el buffer antes de entrenar
    print("Recolectando datos iniciales...")
    # train_collector.collect(n_step=1000, random=True)
    train_collector.collect(n_step=100, random=True)

    # 6. Bucle de Entrenamiento
    print("Iniciando entrenamiento...")
    #result = offpolicy_trainer(
    #    policy, train_collector, test_collector,
    #    max_epoch=50, step_per_epoch=2000, step_per_collect=10,
    #    update_per_step=0.1, episode_per_test=5, batch_size=64,
    #    logger=logger
    #)
    
    # Prueba básica
    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=7,             # Solo 1 época
        step_per_epoch=500,      # Muy pocos pasos por época
        step_per_collect=10,     # Cada 10 pasos simulados, actualiza la red
        update_per_step=0.1,
        episode_per_test=1,      # Solo evalúa 1 episodio al terminar la época
        batch_size=64,
        save_best_fn=save_best_fn,
        logger=logger
    )

    print(f"Entrenamiento finalizado: {result}")
    env.close()
    train_envs.close()
    test_envs.close()

if __name__ == '__main__':
    main()
