import rclpy
import time
from so101_rl.so101_env import SO101Env

def main(args=None):
    print("Iniciando prueba del entorno SO-101...")
    
    # 1. Instanciar el entorno
    env = SO101Env()
    
    # 2. Reset inicial (Por ahora solo lee los primeros datos)
    print("Ejecutando reset()...")
    obs, info = env.reset()
    print(f"Observación inicial recibida:")
    print(f" - Forma de la imagen: {obs['image'].shape}")
    print(f" - Valores de los joints: {obs['joints']}")
    
    # 3. Bucle de prueba con acciones aleatorias
    print("\nEjecutando 5 pasos con acciones aleatorias...")
    for i in range(5):
        # Generar una acción aleatoria válida dentro de nuestro Action Space (-1.0 a 1.0)
        random_action = env.action_space.sample()
        
        # Dar un paso en el entorno
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        print(f"Paso {i+1}:")
        print(f" - Acción enviada: {random_action}")
        print(f" - Nuevos joints: {obs['joints']}")
        time.sleep(1.0) # Pausa para que nos dé tiempo a ver si el robot se mueve en Gazebo
        
    print("\nPrueba finalizada. Cerrando ROS 2...")
    # Apagar el executor y el hilo educadamente
    env.executor.shutdown()
    env.spin_thread.join()
    
    env.node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
