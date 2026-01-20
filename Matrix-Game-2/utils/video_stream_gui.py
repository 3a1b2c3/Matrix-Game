from turtle import mode
import pygame
import numpy as np
from typing import List, Callable

from pipeline.causal_inference import CAMERA_VALUE_MAP,KEYBOARD_IDX

def external_frame_source(width: int, height: int, event_log: List[str]) -> np.ndarray:
    """
    External frame source that generates a random frame where the entire frame has the same random color.
    Additionally, it can use the event log for further processing (if needed).

    Args:
        width (int): Width of the frame.
        height (int): Height of the frame.
        event_log (List[str]): List of recent events.

    Returns:
        np.ndarray: Randomly generated frame as a NumPy array.
    """
    # Generate a single random color
    color = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
    # Fill the entire frame with the same color
    frame = np.tile(color, (height, width, 1))

    # Example usage of event_log (currently unused, but passed for future use)
    # You can process the event_log here if needed.

    return frame

def run_ui(frame_source: Callable[[int, int, List[str]], np.ndarray], width: int = 640, height: int = 480) -> None:
    # Initialize pygame
    pygame.init()

    # Set up display
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Event Recorder")

    # Font for displaying events
    font = pygame.font.Font(None, 24)

    # Clock for controlling frame rate
    clock = pygame.time.Clock()
    fps = 30

    # Separate logs for keyboard and mouse events
    keyboard_event_log: List[str] = []
    mouse_event_log: List[str] = []
    max_events = 4  # Maximum number of events to display per log

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                # Record key press/release events
                event_type = "KEYDOWN" if event.type == pygame.KEYDOWN else "KEYUP"
                keyboard_event_log.append(f"{event_type}: {pygame.key.name(event.key)}")
                if len(keyboard_event_log) > max_events:
                    keyboard_event_log.pop(0)
            elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                # Record mouse button events
                event_type = "MOUSEDOWN" if event.type == pygame.MOUSEBUTTONDOWN else "MOUSEUP"
                mouse_event_log.append(f"{event_type}: Button {event.button} at {event.pos}")
                if len(mouse_event_log) > max_events:
                    mouse_event_log.pop(0)
            elif event.type == pygame.MOUSEMOTION:
                # Record mouse motion events
                mouse_event_log.append(f"MOUSEMOVE: {event.pos}")
                if len(mouse_event_log) > max_events:
                    mouse_event_log.pop(0)

        # Get a frame from the external frame source, passing both logs
        frame = frame_source(width, height, keyboard_event_log + mouse_event_log)

        # Convert the NumPy array to a pygame Surface
        frame_surface = pygame.surfarray.make_surface(frame)

        # Display the frame
        screen.blit(frame_surface, (0, 0))

        # Render the keyboard event log on the screen
        y_offset = 10
        for log in keyboard_event_log:
            text_surface = font.render(f"Keyboard: {log}", True, (255, 255, 255))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 20

        # Render the mouse event log on the screen
        for log in mouse_event_log:
            text_surface = font.render(f"Mouse: {log}", True, (255, 255, 255))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 20

        pygame.display.flip()

        # Cap the frame rate
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    run_ui(external_frame_source)
