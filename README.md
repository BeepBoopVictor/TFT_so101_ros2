<div align="center">
  <h1>SO-101 RL Pick-And-Place</h1>
  <p>Reinforcement Learning para el brazo SO-101 en simulación con Gazebo Fortress e integración ROS2 + Tianshou.</p>
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version"></a>
    <a href="https://docs.ros.org/en/humble/Installation.html"><img src="https://img.shields.io/badge/ROS2-Humble-green.svg" alt="ROS2 Version"></a>
    <a href="https://gazebosim.org/docs/fortress/"><img src="https://img.shields.io/badge/Gazebo-Fortress-orange.svg" alt="Gazebo Fortress"></a>
    <a href="https://tianshou.org/"><img src="https://img.shields.io/badge/Tianshou-RL-blueviolet.svg" alt="Tianshou"></a>
    <a href="https://huggingface.co/docs/lerobot/"><img src="https://img.shields.io/badge/Lerobot-SO--101-ff69b4.svg" alt="Lerobot"></a>
  </p>
  <p><em>Fork de <a href="https://github.com/nimiCurtis/so101_ros2">nimiCurtis/so101_ros2</a> — Trabajo de Fin de Grado</em></p>
</div>

---

## Descripción

Este repositorio es un fork de [so101_ros2](https://github.com/nimiCurtis/so101_ros2) orientado exclusivamente a **aprendizaje por refuerzo en simulación**. El objetivo es entrenar un agente capaz de resolver la tarea de **Pick-And-Place** usando el brazo robótico SO-101 en un entorno simulado con **Gazebo Fortress (Ignition)**, sin necesidad de hardware físico.

El pipeline combina:

- **ROS2 Humble** como middleware de comunicación con el simulador.
- **Gazebo Fortress** (Ignition) como entorno de simulación física.
- **Tianshou** como framework de RL para definir agentes, políticas y bucles de entrenamiento.
- **Gymnasium** para la interfaz estándar del entorno (`gym.Env`).

> **Estado actual:** Entorno Gazebo + integración con Tianshou parcialmente completados. Pipeline de entrenamiento end-to-end en desarrollo activo.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────┐
│                    Agente Tianshou                  │
│         (PPO / SAC / TD3 + Collector + Trainer)     │
└───────────────────┬─────────────────────────────────┘
                    │  step() / reset()
┌───────────────────▼─────────────────────────────────┐
│              Gym Environment Wrapper                │
│          (so101_rl / PickAndPlaceEnv)               │
└───────────────────┬─────────────────────────────────┘
                    │  ROS2 Topics / Services
┌───────────────────▼─────────────────────────────────┐
│          ROS2 Humble (rmw_cyclonedds_cpp)           │
│   /joint_states  /cmd_vel  /model_states  ...       │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│        Gazebo Fortress (Ignition Gazebo)            │
│     Escena Pick-And-Place + SO-101 URDF/SDF         │
└─────────────────────────────────────────────────────┘
```

---

## Dependencias

| Componente | Versión |
|---|---|
| Ubuntu | 22.04 LTS |
| ROS2 | Humble |
| Gazebo | Fortress (Ignition 6) |
| Python | 3.10+ |
| Tianshou | ≥ 1.0 |
| Gymnasium | ≥ 0.29 |
| ros_gz | Humble (bridge ROS2 ↔ Gazebo) |

---

## Instalación

### 1. Instalar ROS2 Humble

Sigue la [guía oficial](https://docs.ros.org/en/humble/Installation.html).

### 2. Instalar Gazebo Fortress y el bridge ROS2

```bash
sudo apt install ros-humble-ros-gz
```

Verifica que `gz sim --version` devuelve `6.x.x`.

### 3. Configurar el entorno Python (Conda)

```bash
conda create -n so101_rl python=3.10
conda activate so101_rl

# Evitar error de libstdc++ con ROS2
conda install -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"
```

### 4. Instalar Tianshou y dependencias de RL

```bash
pip install tianshou gymnasium numpy torch
```

### 5. Clonar e instalar lerobot (descripción del robot)

```bash
git clone https://github.com/nimiCurtis/lerobot.git
cd lerobot
pip install -e ".[all]"
```

### 6. Clonar y compilar el workspace ROS2

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone --recurse-submodules https://github.com/<tu_usuario>/so101_ros2.git
cd so101_ros2
./build.sh
```

### 7. Exportar variables de entorno

```bash
echo export LECONDA_SITE_PACKAGES="<ruta_a_conda>/envs/so101_rl/lib/python3.10/site-packages" >> ~/.bashrc
echo export LEROBOT_SRC="<ruta_a_lerobot>/src/lerobot" >> ~/.bashrc
echo export SO101BRIDGE_INSTALL_SITE_PACKAGES="<ruta_a_ros2_ws>/install/so101_ros2_bridge/lib/python3.10/site-packages/lerobot" >> ~/.bashrc
source ~/.bashrc

ln -s $LEROBOT_SRC $SO101BRIDGE_INSTALL_SITE_PACKAGES
```

---

## Uso

### Lanzar la simulación en Gazebo Fortress

```bash
ros2 launch so101_bringup so101_gazebo.launch.py
```

Esto levanta la escena Pick-And-Place con el SO-101 cargado vía SDF/URDF y el bridge `ros_gz` activo.

### Ejecutar el entrenamiento RL con Tianshou

```bash
conda activate so101_rl
python so101_rl/train.py --algo ppo --env PickAndPlace-v0
```

Los argumentos principales del script de entrenamiento son:

| Argumento | Descripción | Default |
|---|---|---|
| `--algo` | Algoritmo RL (`ppo`, `sac`, `td3`) | `ppo` |
| `--env` | ID del entorno Gymnasium | `PickAndPlace-v0` |
| `--epoch` | Número de épocas | `100` |
| `--step-per-epoch` | Pasos por época | `10000` |
| `--logdir` | Directorio de logs (TensorBoard) | `logs/` |

### Visualizar el entrenamiento

```bash
tensorboard --logdir logs/
```

---

## Estructura del repositorio

```
so101_ros2/
├── so101_bringup/          # Launch files (Gazebo, teleop, RL)
├── so101_description/      # URDF / SDF / meshes del SO-101
├── so101_ros2_bridge/      # Bridge ROS2 ↔ lerobot
├── so101_rl/               # Paquete principal de RL
│   ├── envs/               # Entornos Gymnasium (PickAndPlaceEnv)
│   ├── policies/           # Configuraciones de políticas Tianshou
│   ├── train.py            # Script de entrenamiento
│   └── eval.py             # Script de evaluación
└── so101_gazebo/           # Escenas SDF, mundos y plugins Gazebo
```

---

## Entorno RL: `PickAndPlace-v0`

El entorno implementa la interfaz `gymnasium.Env` y se comunica con Gazebo vía ROS2.

### Espacio de observación

| Campo | Dimensión | Descripción |
|---|---|---|
| `joint_pos` | 6 | Posición de las 6 articulaciones (rad) |
| `joint_vel` | 6 | Velocidad articular (rad/s) |
| `ee_pos` | 3 | Posición del end-effector (x, y, z) |
| `object_pos` | 3 | Posición del objeto a manipular |
| `goal_pos` | 3 | Posición objetivo de la tarea |

### Espacio de acción

Acción continua de dimensión 6 (comandos de posición articular normalizados en `[-1, 1]`).

### Función de recompensa

```
r = -||ee_pos - object_pos|| · w_reach
  + picked · r_pick
  - ||object_pos - goal_pos|| · w_place
  + success · r_success
  - step_penalty
```

---

## Roadmap

- [x] Descripción URDF/SDF del SO-101
- [x] Bridge ROS2 ↔ Gazebo Fortress (`ros_gz`)
- [x] Escena Pick-And-Place completa en Gazebo
- [x] Wrapper `gymnasium.Env` funcional con ROS2
- [x] Integración con Tianshou (Collector + Trainer)
- [ ] Entrenamiento PPO baseline
- [ ] Benchmarks SAC / TD3
- [ ] Evaluación y visualización de políticas entrenadas

---

## Referencia al fork original

Este proyecto parte del trabajo de [nimiCurtis/so101_ros2](https://github.com/nimiCurtis/so101_ros2). Las funcionalidades de teleoperation con hardware real, imitation learning con VLA y la integración con Isaac Sim **no están incluidas en esta rama** y se mantienen en el repositorio upstream.

---

## Licencia

MIT License. Ver [LICENSE](LICENSE) para más detalles.
