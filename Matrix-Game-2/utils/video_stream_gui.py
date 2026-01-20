import pygame
import numpy as np
import random

def generate_random_frame(width: int, height: int) -> np.ndarray:
    """
    Generates a random frame where the entire frame has the same random color.

    Args:
        width (int): Width of the frame.
        height (int): Height of the frame.

    Returns:
        np.ndarray: Randomly generated frame as a NumPy array.
    """
    # Generate a single random color
    color = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
    # Fill the entire frame with the same color
    return np.tile(color, (height, width, 1))

def main():
    # Initialize pygame
    pygame.init()

    # Set up display
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Random Video Stream")

    # Clock for controlling frame rate
    clock = pygame.time.Clock()
    fps = 30

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Generate a random frame
        frame = generate_random_frame(width, height)

        # Convert the NumPy array to a pygame Surface
        frame_surface = pygame.surfarray.make_surface(frame)

        # Display the frame
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    main()
