import sys
import pygame


def handle_events():
    """
    Processa eventos globais do Pygame:
    - fechar janela (X)
    - ESC para sair
    - P para pausar/retomar a simulação
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_p:
                pause_simulation()


def pause_simulation():
    """
    Pausa a simulação até o usuário pressionar P novamente
    ou fechar a janela.
    """
    paused = True
    screen = pygame.display.get_surface()

    # Fonte simples para mensagem de pausa
    font = pygame.font.SysFont(None, 32)
    text = font.render(
        "PAUSADO - pressione P para continuar",
        True,
        (0, 0, 0),
        (200, 200, 200),
    )

    # Centraliza aproximadamente na parte superior
    text_rect = text.get_rect()
    text_rect.topleft = (20, 10)

    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = False

        # Redesenha só o retângulo do texto por cima
        screen.blit(text, text_rect)
        pygame.display.flip()
        pygame.time.wait(100)


def initialize_display(simulator):
    """
    Inicializa a janela do Pygame para exibir o ambiente.

    Parâmetros:
        simulator: objeto que possui o atributo `size`
                   representando o tamanho do grid (NxN).

    Retorna:
        (screen, cell_size): superfície principal e tamanho de cada célula.
    """
    pygame.init()

    # Define o tamanho das células de forma adaptativa ao tamanho do grid
    if simulator.size <= 10:
        cell_size = 60
    elif simulator.size <= 20:
        cell_size = 40
    else:
        cell_size = 20

    width = simulator.size * cell_size
    height = simulator.size * cell_size + 40  # espaço extra para textos

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Simulação de Reinforcement Learning")

    # Pequeno controle de framerate inicial
    clock = pygame.time.Clock()
    clock.tick(60)

    return screen, cell_size


def terminate_display():
    """
    Encerra o Pygame e o programa de forma segura.
    """
    pygame.quit()
    sys.exit()
