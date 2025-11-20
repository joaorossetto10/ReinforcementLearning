from utils import initialize_display, terminate_display
from simulator import Simulator
from learner import LearningAgent

# ============================================================
# Configuração do ambiente
# ------------------------------------------------------------
# ENV_MODE pode ser:
#   "A"       -> grid fixo A (6x6)
#   "B"       -> grid fixo B (6x6)
#   "CUSTOM"  -> grid definido manualmente em CUSTOM_MAP
#   "RANDOM"  -> grid aleatório
# ============================================================
ENV_MODE = "CUSTOM"

# ------------------------------------------------------------
# Grid CUSTOM (usado somente se ENV_MODE == "CUSTOM")
#
# Legenda:
#   R = posição inicial do agente
#   S = saída (porta / área segura)
#   P = presente
#   Z = zumbi
#   # = pedra / obstáculo
#   . = espaço vazio
# ------------------------------------------------------------
CUSTOM_MAP = [
    "R.Z.PP",
    "....ZP",
    ".Z....",
    ".PZ.##",
    ".ZP.ZP",
    "SZZ.PP",
]


def main():
    # --------------------------------------------------------
    # Parâmetros base (usados no modo RANDOM e como referência)
    # --------------------------------------------------------
    grid_size = 6
    num_zombies = 8
    num_presents = 8
    num_obstacles = 2

    # --------------------------------------------------------
    # Inicialização do simulador conforme o modo de ambiente
    # --------------------------------------------------------
    if ENV_MODE == "RANDOM":
        simulator = Simulator(
            grid_size=grid_size,
            num_zombies=num_zombies,
            num_presents=num_presents,
            num_obstacles=num_obstacles,
            load=False,
            layout=None,
        )

    elif ENV_MODE in ("A", "B"):
        simulator = Simulator(
            grid_size=6,
            num_zombies=8,
            num_presents=8,
            num_obstacles=2,
            load=False,
            layout=ENV_MODE,
        )

    elif ENV_MODE == "CUSTOM":
        simulator = Simulator(
            grid_size=len(CUSTOM_MAP),
            num_zombies=0,
            num_presents=0,
            num_obstacles=0,
            load=False,
            layout="CUSTOM",
            custom_map=CUSTOM_MAP,
        )

    else:
        raise ValueError("ENV_MODE inválido. Use 'A', 'B', 'CUSTOM' ou 'RANDOM'.")

    # --------------------------------------------------------
    # Interface gráfica (Pygame)
    # --------------------------------------------------------
    screen, cell_size = initialize_display(simulator)

    # Cria o agente de aprendizado
    agent = LearningAgent(simulator)

    # --------------------------------------------------------
    # Treinamento (não estamos salvando Q-Table em arquivo)
    # --------------------------------------------------------
    agent.train(screen, cell_size)

    # --------------------------------------------------------
    # Teste do agente treinado (política greedy)
    # --------------------------------------------------------
    status, collected_items, steps = agent.test(screen, cell_size)
    total_reward = simulator.total_reward

    print("---------------------")
    print(f"Modo de ambiente: {ENV_MODE}")
    print(f"Status: {status}")
    print(f"Presentes coletados: {len(collected_items)} de {simulator.num_presents}")
    print(f"Quantidade de Passos: {steps}")
    print(f"Recompensa total acumulada: {total_reward}")
    print("---------------------")

    # Encerra Pygame
    terminate_display()


if __name__ == "__main__":
    main()
