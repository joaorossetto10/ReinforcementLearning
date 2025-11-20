import random
import pickle
import numpy as np
import pygame


class Simulator:
    """
    Simulador do ambiente de grid com:
    - Zumbis
    - Presentes
    - Obstáculos
    - Posição inicial (R)
    - Posição final / saída (S)
    """

    def __init__(
        self,
        grid_size=10,
        num_zombies=8,
        num_presents=5,
        num_obstacles=3,
        load=False,
        layout=None,
        custom_map=None,
    ):
        # Configuração básica do grid
        self.size = grid_size
        self.grid = np.zeros((self.size, self.size), dtype=int)

        # Posições padrão (em layouts aleatórios)
        self.start_position = (0, 0)
        self.goal_position = (self.size - 1, self.size - 1)
        self.num_presents = num_presents

        # --------------------------------------------------------
        # TIPOS DE AMBIENTE
        #   - layout == "A" ou "B"  → grids fixos
        #   - layout == "CUSTOM"    → usa custom_map (lista de strings)
        #   - layout == None        → modo aleatório (com save/load)
        # --------------------------------------------------------
        if layout == "CUSTOM" and custom_map is not None:
            self._apply_custom_map(custom_map)

        elif layout is not None:
            self._apply_layout(layout)

        else:
            # Modo aleatório (com persistência em grid.pkl)
            grid_data = self.load_grid() if load else None

            if grid_data:
                (
                    self.zombie_positions,
                    self.present_positions,
                    self.obstacle_positions,
                ) = grid_data
            else:
                self.zombie_positions = self.place_random(num_zombies)
                self.present_positions = self.place_random(
                    num_presents,
                    exclude=self.zombie_positions,
                )
                self.obstacle_positions = self.place_random(
                    num_obstacles,
                    exclude=self.zombie_positions + self.present_positions,
                )
                self.save_grid()

        # Estado dinâmico de um episódio
        self.collected_presents = set()
        self.current_position = self.start_position
        self.total_reward = 0
        self.steps = 0

        # Marca elementos no grid (em CUSTOM já vem pronto, mas reforçamos)
        for i, j in self.zombie_positions:
            self.grid[i][j] = 1  # zumbi
        for i, j in self.present_positions:
            self.grid[i][j] = 2  # presente
        for i, j in self.obstacle_positions:
            self.grid[i][j] = 3  # obstáculo

    # ------------------------------------------------------------------ #
    # CONFIGURAÇÃO DOS LAYOUTS FIXOS                                     #
    # ------------------------------------------------------------------ #

    def _apply_layout(self, layout: str) -> None:
        """
        Define layouts fixos para o trabalho (A ou B).
        Foram pensados para grid 6x6.
        """
        if self.size != 6:
            raise ValueError(
                "Layouts fixos foram definidos para grid 6x6. "
                "Ajuste o grid_size ou os layouts."
            )

        if layout == "A":
            self.zombie_positions = [
                (1, 4), (1, 1), (5, 4), (3, 3),
                (2, 4), (4, 1), (2, 1), (3, 5),
            ]
            self.present_positions = [
                (0, 1), (0, 2), (1, 2), (3, 4),
                (4, 0), (4, 3), (5, 1), (5, 2),
            ]
            self.obstacle_positions = [
                (3, 0), (3, 1),
            ]

        elif layout == "B":
            self.zombie_positions = [
                (0, 2), (1, 4), (2, 1), (3, 2),
                (4, 1), (4, 4), (5, 1), (5, 2),
            ]
            self.present_positions = [
                (0, 4), (0, 5), (1, 5), (3, 1),
                (4, 2), (5, 4), (3, 0), (4, 5),
            ]
            self.obstacle_positions = [
                (3, 4), (3, 5),
            ]
        else:
            raise ValueError("Layout desconhecido. Use 'A', 'B', 'CUSTOM' ou None.")

        # Garante que início e fim estejam livres
        for pos in [self.start_position, self.goal_position]:
            if pos in self.zombie_positions:
                self.zombie_positions.remove(pos)
            if pos in self.present_positions:
                self.present_positions.remove(pos)
            if pos in self.obstacle_positions:
                self.obstacle_positions.remove(pos)

        # Atualiza contagem de presentes
        self.num_presents = len(self.present_positions)

    # ------------------------------------------------------------------ #
    # GERADOR ALEATÓRIO                                                  #
    # ------------------------------------------------------------------ #

    def place_random(self, num_items: int, exclude=None):
        """Gera posições aleatórias no grid, evitando start, goal e lista exclude."""
        if exclude is None:
            exclude = []

        positions = []
        while len(positions) < num_items:
            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)

            if (
                (i, j) not in positions
                and (i, j) not in exclude
                and (i, j) != self.start_position
                and (i, j) != self.goal_position
            ):
                positions.append((i, j))

        return positions

    # ------------------------------------------------------------------ #
    # MAPA CUSTOM                                                        #
    # ------------------------------------------------------------------ #

    def _apply_custom_map(self, grid_map):
        """
        Carrega um grid definido manualmente.
        Caracteres aceitos por célula:
          - 'R' : início (agent / robot)
          - 'S' : saída
          - 'Z' : zumbi
          - 'P' : presente
          - '#' : obstáculo
          - '.' : vazio
        """
        self.zombie_positions = []
        self.present_positions = []
        self.obstacle_positions = []

        self.size = len(grid_map)
        self.grid = np.zeros((self.size, self.size), dtype=int)

        for i, row in enumerate(grid_map):
            for j, cell in enumerate(row):
                if cell == "R":
                    self.start_position = (i, j)
                elif cell == "S":
                    self.goal_position = (i, j)
                elif cell == "Z":
                    self.zombie_positions.append((i, j))
                elif cell == "P":
                    self.present_positions.append((i, j))
                elif cell == "#":
                    self.obstacle_positions.append((i, j))
                # '.' → vazio

        self.num_presents = len(self.present_positions)

        # Marca no grid
        for x, y in self.zombie_positions:
            self.grid[x][y] = 1
        for x, y in self.present_positions:
            self.grid[x][y] = 2
        for x, y in self.obstacle_positions:
            self.grid[x][y] = 3

    # ------------------------------------------------------------------ #
    # LOOP DE EPISÓDIO                                                   #
    # ------------------------------------------------------------------ #

    def reset(self):
        """Reinicia o ambiente para um novo episódio."""
        self.current_position = self.start_position
        self.collected_presents.clear()
        self.total_reward = 0
        self.steps = 0
        return self.current_position, tuple(self.collected_presents)

    def step(self, action: int):
        """
        Executa uma ação no ambiente.

        Ações:
            0 - CIMA
            1 - BAIXO
            2 - ESQUERDA
            3 - DIREITA

        Retorna:
            next_position, collected_presents, reward, done, status
        """
        status = ""
        i, j = self.current_position

        # Movimento
        if action == 0:      # CIMA
            i = max(i - 1, 0)
        elif action == 1:    # BAIXO
            i = min(i + 1, self.size - 1)
        elif action == 2:    # ESQUERDA
            j = max(j - 1, 0)
        elif action == 3:    # DIREITA
            j = min(j + 1, self.size - 1)

        # Impede andar sobre obstáculos (pedras)
        if (i, j) in self.obstacle_positions:
            i, j = self.current_position

        self.current_position = (i, j)

        # ----------------- REGRAS DE RECOMPENSA ----------------- #
        if self.current_position in self.zombie_positions:
            reward = -10
            done = True
            status = "ATACADO POR ZUMBI"

        elif (
            self.current_position in self.present_positions
            and self.current_position not in self.collected_presents
        ):
            self.collected_presents.add(self.current_position)
            reward = +10
            done = False
            status = "COLETOU SUPRIMENTO"

        elif self.current_position == self.goal_position:
            # Só pode escapar depois de pegar todos os presentes
            if len(self.collected_presents) == self.num_presents:
                reward = +20
                done = True
                status = "ALCANÇOU ÁREA SEGURA"
            else:
                reward = -1
                done = False
                status = "PORTA BLOQUEADA (FALTAM PRESENTES)"

        else:
            reward = -1
            done = False
            status = "ANDANDO"

        self.total_reward += reward
        self.steps += 1

        return self.current_position, tuple(self.collected_presents), reward, done, status

    # ------------------------------------------------------------------ #
    # RENDERIZAÇÃO                                                       #
    # ------------------------------------------------------------------ #

    def _load_sprites(self, cell_size: int) -> None:
        """Carrega sprites dos elementos do grid."""

        self.sprites = {"cell_size": cell_size}

        def make_surface(color):
            surf = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
            surf.fill(color)
            return surf

        def load_sprite(filename, fallback_color):
            try:
                img = pygame.image.load(filename).convert_alpha()
                img = pygame.transform.smoothscale(img, (cell_size, cell_size))
                return img
            except Exception:
                return make_surface(fallback_color)

        self.sprites["robot"] = load_sprite("assets/images/agent.png", (0, 0, 255))
        self.sprites["goal"] = load_sprite("assets/images/goal.png", (0, 255, 0))
        self.sprites["zombie"] = load_sprite("assets/images/zombie.png", (255, 0, 0))
        self.sprites["present"] = load_sprite("assets/images/present.png", (255, 215, 0))
        self.sprites["obstacle"] = load_sprite("assets/images/obstacle.png", (100, 100, 100))
        self.sprites["empty"] = make_surface((0, 0, 0, 0))  # transparente

    def render(self, screen, cell_size: int = 60) -> None:
        """Desenha o grid e os elementos na tela do Pygame."""
        if not hasattr(self, "sprites") or self.sprites.get("cell_size") != cell_size:
            self._load_sprites(cell_size)

        screen.fill((100, 200, 120))

        # Desenha cada célula
        for i in range(self.size):
            for j in range(self.size):
                x = j * cell_size
                y = i * cell_size
                sprite = self.sprites["empty"]

                if (i, j) == self.current_position:
                    sprite = self.sprites["robot"]
                elif (i, j) == self.goal_position:
                    sprite = self.sprites["goal"]
                elif self.grid[i][j] == 1:
                    sprite = self.sprites["zombie"]
                elif self.grid[i][j] == 2 and (i, j) not in self.collected_presents:
                    sprite = self.sprites["present"]
                elif self.grid[i][j] == 3:
                    sprite = self.sprites["obstacle"]

                screen.blit(sprite, (x, y))

        # Linhas do grid
        for k in range(self.size + 1):
            # horizontais
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (0, k * cell_size),
                (self.size * cell_size, k * cell_size),
            )
            # verticais
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (k * cell_size, 0),
                (k * cell_size, self.size * cell_size),
            )

        # Info de recompensa/passos
        font = pygame.font.SysFont(None, 24)
        info_text = f"Recompensa: {self.total_reward:.1f}  |  Passos: {self.steps}"
        text_surface = font.render(info_text, True, (0, 0, 0))
        screen.blit(text_surface, (5, 5))

        pygame.display.flip()

    # ------------------------------------------------------------------ #
    # PERSISTÊNCIA DO GRID                                               #
    # ------------------------------------------------------------------ #

    def save_grid(self) -> None:
        """Salva a configuração atual do grid em grid.pkl."""
        print("---------------------------------")
        print("SALVANDO O GRID..................")
        print("---------------------------------")

        grid_data = {
            "zombie_positions": self.zombie_positions,
            "present_positions": self.present_positions,
            "obstacle_positions": self.obstacle_positions,
        }

        with open("grid.pkl", "wb") as file:
            pickle.dump(grid_data, file)

    def load_grid(self):
        """Carrega configuração salva de grid.pkl, se existir."""
        try:
            print("---------------------------------")
            print("CARREGANDO O GRID................")
            print("---------------------------------")

            with open("grid.pkl", "rb") as file:
                grid_data = pickle.load(file)

            return (
                grid_data["zombie_positions"],
                grid_data["present_positions"],
                grid_data["obstacle_positions"],
            )
        except FileNotFoundError:
            return None
