import random
from collections import deque

import numpy as np
import pygame

from utils import handle_events


class LearningAgent:
    """
    Agente que aprende uma política de coleta de suprimentos
    e fuga usando Q-Learning em um grid com zumbis, obstáculos e porta.
    """

    def __init__(self, simulator):
        # Hiperparâmetros de treinamento
        self.total_episodes = 11000          # número máximo de episódios
        self.max_steps = simulator.size * 10  # limite de passos por episódio
        self.learning_rate = 0.1            # alfa
        self.discount_factor = 0.99         # gamma
        self.exploration_rate = 1.0         # epsilon inicial
        self.min_exploration = 0.01         # epsilon mínimo
        self.exploration_decay = 0.001      # taxa de decaimento do epsilon

        # Ambiente
        self.simulator = simulator
        self.grid_size = simulator.size
        self.items_to_collect = list(simulator.present_positions)  # ordem fixa dos presentes

        # Tabela Q: [linha][coluna][máscara_itens][ação]
        num_items = len(self.items_to_collect)
        self.q_table = np.zeros(
            (self.grid_size, self.grid_size, 2**num_items, 4),
            dtype=float,
        )

    # --------------------------------------------------------------------- #
    # Utilidades internas
    # --------------------------------------------------------------------- #

    def _items_to_index(self, collected_items):
        """
        Converte o conjunto de itens coletados em um índice inteiro
        usando máscara binária com base em self.items_to_collect.
        """
        collected_set = set(collected_items)
        bits = [
            "1" if (i, j) in collected_set else "0"
            for (i, j) in self.items_to_collect
        ]
        if not bits:
            return 0
        return int("".join(bits), 2)

    def _simulate_move(self, state, action):
        """
        Simula a próxima posição após uma ação,
        respeitando bordas e obstáculos (não altera o estado real).
        """
        i, j = state
        size = self.simulator.size

        # 0: CIMA, 1: BAIXO, 2: ESQUERDA, 3: DIREITA
        if action == 0:
            i = max(i - 1, 0)
        elif action == 1:
            i = min(i + 1, size - 1)
        elif action == 2:
            j = max(j - 1, 0)
        elif action == 3:
            j = min(j + 1, size - 1)

        # Se bater em obstáculo, permanece na mesma célula
        if (i, j) in self.simulator.obstacle_positions:
            return state

        return (i, j)

    def _bfs_to_goal(self, start):
        """
        Busca em largura até a porta (goal), evitando zumbis e obstáculos.
        Retorna lista de ações [0..3]. Se não houver caminho, retorna [].
        (Mantido como utilitário extra; não usado diretamente no Q-Learning.)
        """
        goal = self.simulator.goal_position
        size = self.simulator.size
        zombies = set(self.simulator.zombie_positions)

        if start == goal:
            return []

        visited = set([start])
        parent = {}  # pos -> (pos_anterior, ação)
        queue = deque([start])

        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for action in range(4):
                nxt = self._simulate_move(current, action)

                # Ignora movimentos que não saem do lugar
                if nxt == current:
                    continue

                # Evita zumbis
                if nxt in zombies:
                    continue

                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = (current, action)
                    queue.append(nxt)

        if goal not in parent:
            return []

        # Reconstrói o caminho de goal até start
        actions = []
        cur = goal
        while cur != start:
            prev, act = parent[cur]
            actions.append(act)
            cur = prev

        actions.reverse()
        return actions

    # --------------------------------------------------------------------- #
    # Política e Q-Learning
    # --------------------------------------------------------------------- #

    def choose_action(self, state, collected_items):
        """
        Política epsilon-greedy usada no treino:
        explora com probabilidade epsilon, caso contrário escolhe a melhor ação.
        """
        item_index = self._items_to_index(collected_items)

        # Exploração
        if random.uniform(0.0, 1.0) < self.exploration_rate:
            return random.randint(0, 3)  # 0:CIMA, 1:BAIXO, 2:ESQ, 3:DIR

        # Exploitation (ação com maior valor Q)
        return int(np.argmax(self.q_table[state[0], state[1], item_index]))

    def train(self, screen, cell_size):
        """
        Treino do agente via Q-Learning.
        """
        print("---------------------------------")
        print("TREINANDO O AGENTE...........")
        rewards_history = []
        window_size = 1000

        for episode in range(self.total_episodes + 1):
            state, collected_items = self.simulator.reset()
            done = False
            steps = 0
            episode_reward = 0.0

            while not done and steps < self.max_steps:
                handle_events()

                action = self.choose_action(state, collected_items)
                next_state, next_items, reward, done, status = self.simulator.step(
                    action
                )

                current_item_index = self._items_to_index(collected_items)
                next_item_index = self._items_to_index(next_items)

                # Q atual
                old_value = self.q_table[
                    state[0], state[1], current_item_index, action
                ]

                # Target Bellman
                if done:
                    target = reward
                else:
                    next_max = np.max(
                        self.q_table[next_state[0], next_state[1], next_item_index]
                    )
                    target = reward + self.discount_factor * next_max

                # Atualização Q-Learning
                new_value = old_value + self.learning_rate * (target - old_value)
                self.q_table[
                    state[0], state[1], current_item_index, action
                ] = new_value

                state, collected_items = next_state, next_items
                episode_reward += reward
                steps += 1

            rewards_history.append(episode_reward)

            # Atualiza epsilon (exploração)
            self.exploration_rate = max(
                self.min_exploration,
                self.exploration_rate * (1.0 - self.exploration_decay),
            )

            # Log a cada 1000 episódios
            if episode % 1000 == 0:
                if rewards_history:
                    window = rewards_history[-window_size:]
                    avg_reward = sum(window) / len(window)
                else:
                    avg_reward = 0.0

                print(
                    f"Episódio: {episode:5d} | "
                    f"Recompensa média (últ. {window_size}): {avg_reward:6.2f} | "
                    f"Epsilon: {self.exploration_rate:.3f}"
                )

                # Renderiza para visualização durante o treino
                self.simulator.render(screen, cell_size)
                pygame.time.wait(200)

    # --------------------------------------------------------------------- #
    # Teste (política greedy)
    # --------------------------------------------------------------------- #

    def test(self, screen, cell_size):
        """
        Executa o agente no ambiente usando apenas a política aprendida (greedy).
        """
        state, collected_items = self.simulator.reset()
        done = False
        steps = 0
        total_reward = 0.0
        status = "ANDANDO"

        action_names = {
            0: "CIMA",
            1: "BAIXO",
            2: "ESQUERDA",
            3: "DIREITA",
        }

        print("TESTANDO O AGENTE (passo a passo):")
        print("------------------------------------------------------")

        while not done and steps < self.max_steps:
            handle_events()

            item_index = self._items_to_index(collected_items)
            q_values = self.q_table[state[0], state[1], item_index]
            action = int(np.argmax(q_values))

            # Executa ação
            next_state, next_items, reward, done, new_status = self.simulator.step(
                action
            )
            total_reward += reward
            steps += 1

            if new_status:
                status = new_status

            action_name = action_names.get(action, str(action))
            print(
                f"Passo {steps:02d}: {action_name:<7} → {status:<40} "
                f"Recompensa: {reward:+5.1f} | Total: {total_reward:+5.1f}"
            )

            self.simulator.render(screen, cell_size)
            pygame.time.wait(300)

            state, collected_items = next_state, next_items

        if not done and steps >= self.max_steps:
            status = "LIMITE DE PASSOS / SEM SOLUÇÃO"

        print("------------------------------------------------------")
        print(f"Status final: {status}")
        print(
            f"Presentes coletados: {len(collected_items)} "
            f"de {self.simulator.num_presents}"
        )
        print(f"Passos executados: {steps}")
        print(f"Recompensa total acumulada: {total_reward:.0f}")
        print("------------------------------------------------------")

        return status, collected_items, steps
