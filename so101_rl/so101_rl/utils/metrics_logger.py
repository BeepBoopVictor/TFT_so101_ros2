# Sistema de Métricas Extendido para HER + SAC - SO101
# Autor: Víctor Gil
# Registra métricas por paso, episodio y época para análisis posterior y visualización.

import os
import csv
import time
import numpy as np
from collections import defaultdict


class MetricsLogger:
    """
    Registra métricas detalladas durante el entrenamiento HER+SAC.

    Genera tres CSVs:
      - steps.csv    → una fila por paso de simulación
      - episodes.csv → una fila por episodio completado
      - epochs.csv   → una fila por época (resumen agregado)
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # --- Archivos CSV ---
        self._init_csv('steps.csv',    ['epoch', 'episode', 'step', 'timestamp',
                                        'distance_to_cube', 'reward',
                                        'gripper_x', 'gripper_y', 'gripper_z',
                                        'cube_x', 'cube_y', 'cube_z'])

        self._init_csv('episodes.csv', ['epoch', 'episode', 'total_steps',
                                        'total_reward', 'min_distance',
                                        'final_distance', 'success',
                                        'duration_sec'])

        self._init_csv('epochs.csv',   ['epoch', 'n_episodes', 'mean_reward',
                                        'std_reward', 'mean_min_distance',
                                        'std_min_distance', 'mean_steps',
                                        'success_rate', 'mean_final_distance'])

        # --- Estado interno ---
        self.current_epoch   = 0
        self.current_episode = 0
        self.global_step     = 0

        # Buffer del episodio en curso
        self._ep_reset()

        # Buffer de la época en curso
        self._epoch_episodes = []

    # ------------------------------------------------------------------ helpers
    def _init_csv(self, filename: str, headers: list):
        path = os.path.join(self.log_dir, filename)
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
        setattr(self, f'_path_{filename.replace(".","_")}', path)

    def _append_csv(self, filename: str, row: list):
        path = os.path.join(self.log_dir, filename)
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def _ep_reset(self):
        self._ep_rewards   = []
        self._ep_distances = []
        self._ep_steps     = 0
        self._ep_start     = time.time()

    # ------------------------------------------------------------------ API pública

    def new_epoch(self):
        """Llamar al inicio de cada época."""
        self.current_epoch += 1
        self._epoch_episodes = []

    def log_step(self, reward: float, distance: float,
                 gripper_pos: np.ndarray, cube_pos: np.ndarray):
        """
        Llamar en cada paso de simulación (dentro de step()).

        Parámetros
        ----------
        reward      : recompensa recibida en este paso
        distance    : distancia pinza-cubo en metros
        gripper_pos : array [x, y, z] posición absoluta de la pinza
        cube_pos    : array [x, y, z] posición absoluta del cubo
        """
        self.global_step    += 1
        self._ep_steps      += 1
        self._ep_rewards.append(reward)
        self._ep_distances.append(distance)

        gx, gy, gz = gripper_pos
        cx, cy, cz = cube_pos

        self._append_csv('steps.csv', [
            self.current_epoch, self.current_episode, self.global_step,
            round(time.time(), 3),
            round(distance, 5), round(reward, 5),
            round(float(gx), 5), round(float(gy), 5), round(float(gz), 5),
            round(float(cx), 5), round(float(cy), 5), round(float(cz), 5),
        ])

    def log_episode_end(self, success: bool):
        """
        Llamar cuando termina un episodio (terminated o truncated).

        Parámetros
        ----------
        success : True si el agente alcanzó el objetivo
        """
        self.current_episode += 1
        duration     = time.time() - self._ep_start
        total_reward = sum(self._ep_rewards)
        min_dist     = min(self._ep_distances) if self._ep_distances else 0.0
        final_dist   = self._ep_distances[-1]  if self._ep_distances else 0.0

        row = {
            'epoch':          self.current_epoch,
            'episode':        self.current_episode,
            'total_steps':    self._ep_steps,
            'total_reward':   round(total_reward, 4),
            'min_distance':   round(min_dist, 5),
            'final_distance': round(final_dist, 5),
            'success':        int(success),
            'duration_sec':   round(duration, 2),
        }
        self._append_csv('episodes.csv', list(row.values()))
        self._epoch_episodes.append(row)
        self._ep_reset()

    def log_epoch_end(self):
        """
        Llamar al final de cada época para guardar el resumen agregado.
        Devuelve un dict con las métricas de la época para poder logearlas en TensorBoard.
        """
        eps = self._epoch_episodes
        if not eps:
            return {}

        rewards       = [e['total_reward']   for e in eps]
        min_dists     = [e['min_distance']   for e in eps]
        final_dists   = [e['final_distance'] for e in eps]
        steps         = [e['total_steps']    for e in eps]
        successes     = [e['success']        for e in eps]

        summary = {
            'epoch':              self.current_epoch,
            'n_episodes':         len(eps),
            'mean_reward':        round(float(np.mean(rewards)),     4),
            'std_reward':         round(float(np.std(rewards)),      4),
            'mean_min_distance':  round(float(np.mean(min_dists)),   5),
            'std_min_distance':   round(float(np.std(min_dists)),    5),
            'mean_steps':         round(float(np.mean(steps)),       2),
            'success_rate':       round(float(np.mean(successes)),   4),
            'mean_final_distance':round(float(np.mean(final_dists)), 5),
        }
        self._append_csv('epochs.csv', list(summary.values()))
        return summary