import os
import torch
import numpy as np
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Batch

from so101_rl.envs.so101_env import SO101Env

def main():

    """
    	función para evaluar el entrenamiento por medio del fichero .pth
    """

    print("Iniciando entorno para evaluación...")
    env = SO101Env()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. misma arquitectura de red
    net_a = Net(state_shape, hidden_sizes=, device=device)
    actor = ActorProb(net_a, action_shape, max_action=env.action_space.high, device=device, unbounded=True).to(device)
    
    net_c1 = Net(state_shape, action_shape, hidden_sizes=, concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    
    net_c2 = Net(state_shape, action_shape, hidden_sizes=, concat=True, device=device)
    critic2 = Critic(net_c2, device=device).to(device)

    policy = SACPolicy(
        actor, torch.optim.Adam(actor.parameters()), 
        critic1, torch.optim.Adam(critic1.parameters()), 
        critic2, torch.optim.Adam(critic2.parameters()), 
        action_space=env.action_space
    )

    # 2. Cargar los pesos guardados
    model_path = os.path.join("log", "sac_so101_phase1", "policy.pth")
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"¡Modelo cargado exitosamente desde {model_path}!")
    else:
        print(f"ADVERTENCIA: No se encontró {model_path}. Se usará una política aleatoria.")

    # Se activa el modo evoluación para apagar la exploración y ver el comportamiento real aprendido
    policy.eval() 

    # 3. Bucle de evaluación
    print("\nEjecutando episodio de prueba...")
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        batch = Batch(obs=[obs], info=info)
        
        act_res = policy(batch)
        action = act_res.act.detach().cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        print(f"Paso {step_count:03d} | Recompensa: {reward:.2f} | Distancia al cubo: {info.get('distance_to_cube', 0):.3f}m")

    print(f"\nEvaluación finalizada. Pasos: {step_count}, Recompensa total: {total_reward:.2f}")
    env.close()

if __name__ == '__main__':
    main()
