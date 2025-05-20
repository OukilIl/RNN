import pygame
import cv2
import sys
import numpy as np
from sokoban_game import SokobanGame, TILE_SIZE
from webcam_gesture_recognition import HandGestureRecognizer

# Initialize Pygame
pygame.init()

# Constants
CAMERA_WIDTH = 240
CAMERA_HEIGHT = 180
CAMERA_MARGIN = 20
INFO_HEIGHT = 120
PADDING = 15
FPS = 30
BG_COLOR = (40, 40, 40)
TEXT_COLOR = (220, 220, 220)
PANEL_COLOR = (60, 60, 60)
BORDER_COLOR = (100, 100, 100)

def main():
    # Initialize the Sokoban game
    game = SokobanGame()
    
    # Calculate display dimensions based on the game size
    game_width = len(game.level[0]) * TILE_SIZE
    game_height = len(game.level) * TILE_SIZE
    
    # Make the screen size proportional to game size with enough space for UI elements
    screen_width = game_width + CAMERA_WIDTH + 3*PADDING
    screen_height = max(game_height, CAMERA_HEIGHT + INFO_HEIGHT + 2*PADDING) + PADDING
    
    # Initialize screen
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Hand Gesture Controlled Sokoban")
    
    # Initialize the hand gesture recognizer
    try:
        recognizer = HandGestureRecognizer("hand_gesture_model.h5")
    except Exception as e:
        print(f"Error loading gesture model: {e}")
        print("Falling back to keyboard controls only")
        recognizer = None
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        recognizer = None
    
    # Game clock
    clock = pygame.time.Clock()
    
    # Fonts for info text
    title_font = pygame.font.SysFont(None, 28)
    info_font = pygame.font.SysFont(None, 24)
    
    # Main game loop
    running = True
    while running:
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.move_player(0, -1)
                elif event.key == pygame.K_DOWN:
                    game.move_player(0, 1)
                elif event.key == pygame.K_LEFT:
                    game.move_player(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    game.move_player(1, 0)
                elif event.key == pygame.K_r:
                    game.reset_level()
                elif event.key == pygame.K_n and game.is_completed:
                    game.next_level()
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Process webcam feed and gestures
        gesture_text = "None detected"
        control_text = "Keyboard only"
        
        if recognizer and cap.isOpened():
            # Read a frame from the webcam
            ret, frame = cap.read()
            if ret:
                # Mirror the frame for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Make a prediction
                label, confidence, game_control = recognizer.predict(frame)
                
                # Update text
                gesture_text = f"{label} ({confidence:.2f})"
                
                # Apply game control if a gesture is detected
                if game_control:
                    control_text = f"{game_control}"
                    if game_control == "UP":
                        game.move_player(0, -1)
                    elif game_control == "DOWN":
                        game.move_player(0, 1)
                    elif game_control == "LEFT":
                        game.move_player(-1, 0)
                    elif game_control == "RIGHT":
                        game.move_player(1, 0)
                
                # Convert frame to Pygame surface for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (CAMERA_WIDTH, CAMERA_HEIGHT))
                camera_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            else:
                # If frame reading failed
                camera_surface = pygame.Surface((CAMERA_WIDTH, CAMERA_HEIGHT))
                camera_surface.fill((0, 0, 0))  # Black screen
        else:
            # If webcam or recognizer not available
            camera_surface = pygame.Surface((CAMERA_WIDTH, CAMERA_HEIGHT))
            camera_surface.fill((0, 0, 0))  # Black screen
        
        # Clear the screen with background color
        screen.fill(BG_COLOR)
        
        # Calculate layout positions
        game_pos_x = PADDING
        game_pos_y = PADDING
        right_panel_x = game_width + 2*PADDING
        right_panel_width = CAMERA_WIDTH + PADDING
        
        # Draw game to its own surface
        game_surface = pygame.Surface((game_width, game_height))
        game.screen = game_surface  # Temporarily override game's screen
        game.draw()
        game.screen = screen  # Restore original screen
        
        # Create a background panel for the game
        pygame.draw.rect(
            screen, 
            PANEL_COLOR, 
            (game_pos_x - 5, game_pos_y - 5, game_width + 10, game_height + 10),
            0,
            5  # Rounded corners
        )
        
        # Blit game surface to main screen
        screen.blit(game_surface, (game_pos_x, game_pos_y))
        
        # Create a panel for the right side elements
        pygame.draw.rect(
            screen, 
            PANEL_COLOR, 
            (right_panel_x, PADDING, right_panel_width, screen_height - 2*PADDING),
            0,
            5  # Rounded corners
        )
        
        # Draw camera title
        camera_title = title_font.render("Camera Feed", True, TEXT_COLOR)
        screen.blit(camera_title, (right_panel_x + (right_panel_width - camera_title.get_width()) // 2, PADDING + 10))
        
        # Draw webcam feed with a border
        camera_pos_x = right_panel_x + (right_panel_width - CAMERA_WIDTH) // 2
        camera_pos_y = PADDING + 40
        
        # Border for camera
        pygame.draw.rect(
            screen, 
            BORDER_COLOR, 
            (camera_pos_x - 2, camera_pos_y - 2, CAMERA_WIDTH + 4, CAMERA_HEIGHT + 4),
            0,
            3  # Rounded corners
        )
        
        # Camera feed
        screen.blit(camera_surface, (camera_pos_x, camera_pos_y))
        
        # Info section
        info_y = camera_pos_y + CAMERA_HEIGHT + 20
        
        # Draw info panel titles and values
        gesture_title = info_font.render("Detected Gesture:", True, TEXT_COLOR)
        gesture_value = info_font.render(gesture_text, True, TEXT_COLOR)
        
        control_title = info_font.render("Control Action:", True, TEXT_COLOR)
        control_value = info_font.render(control_text, True, TEXT_COLOR)
        
        # Position the info text
        screen.blit(gesture_title, (right_panel_x + 10, info_y))
        screen.blit(gesture_value, (right_panel_x + 10, info_y + 25))
        
        screen.blit(control_title, (right_panel_x + 10, info_y + 60))
        screen.blit(control_value, (right_panel_x + 10, info_y + 85))
        
        # Draw controls help panel at the bottom
        help_panel_y = screen_height - PADDING - 35
        pygame.draw.rect(
            screen, 
            PANEL_COLOR, 
            (PADDING, help_panel_y, screen_width - 2*PADDING, 35),
            0,
            5  # Rounded corners
        )
        
        # Draw controls help
        help_text = "Arrow keys: Move | R: Reset | N: Next level | ESC: Quit"
        controls_surf = info_font.render(help_text, True, TEXT_COLOR)
        controls_x = (screen_width - controls_surf.get_width()) // 2
        screen.blit(controls_surf, (controls_x, help_panel_y + 10))
        
        # Update the display
        pygame.display.flip()
        
        # Cap the FPS
        clock.tick(FPS)
    
    # Clean up resources
    if cap.isOpened():
        cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 