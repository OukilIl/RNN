import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Game constants
TILE_SIZE = 64
GAME_TITLE = "Hand Gesture Sokoban"
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FLOOR_COLOR = (220, 220, 220)
WALL_COLOR = (120, 120, 120)
PLAYER_COLOR = (0, 128, 255)
BOX_COLOR = (165, 42, 42)
TARGET_COLOR = (0, 255, 0)
BOX_ON_TARGET_COLOR = (255, 165, 0)

# Game elements
WALL = '#'
FLOOR = ' '
PLAYER = 'P'
BOX = 'B'
TARGET = 'T'
BOX_ON_TARGET = 'X'
PLAYER_ON_TARGET = 'Y'

# Example game levels (you can add more)
LEVELS = [
    [
        "##########",
        "#        #",
        "#  BTTT  #",
        "#  B     #",
        "#  B  P  #",
        "#        #",
        "##########"
    ],
    [
        "############",
        "#          #",
        "#  #  ###  #",
        "#  # BP    #",
        "#  # B###  #",
        "#  ###     #",
        "#    T     #",
        "#    T     #",
        "#    T     #",
        "#          #",
        "############"
    ],
    [
        "##############",
        "#            #",
        "#  TTTTT     #",
        "#            #",
        "#  #####     #",
        "#            #",
        "#  BBBBB     #",
        "#            #",
        "#     P      #",
        "#            #",
        "##############"
    ]
]

class SokobanGame:
    def __init__(self, level_idx=0):
        self.level_idx = level_idx
        self.load_level(level_idx)
        
        # Set up the display
        self.width = len(self.level[0]) * TILE_SIZE
        self.height = len(self.level) * TILE_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(GAME_TITLE)
        
        # Set up the clock
        self.clock = pygame.time.Clock()
        
        # Game state
        self.moves = 0
        self.is_completed = False
        
    def load_level(self, level_idx):
        """Load a level from the LEVELS list"""
        if 0 <= level_idx < len(LEVELS):
            level_data = LEVELS[level_idx]
            
            # Convert level data to a 2D array
            self.level = []
            for row in level_data:
                self.level.append(list(row))
            
            # Find the player position
            self.player_pos = None
            for y, row in enumerate(self.level):
                for x, cell in enumerate(row):
                    if cell == PLAYER or cell == PLAYER_ON_TARGET:
                        self.player_pos = (x, y)
                        break
                if self.player_pos:
                    break
            
            if not self.player_pos:
                print("Error: No player position found in the level.")
                sys.exit(1)
        else:
            print(f"Error: Invalid level index {level_idx}.")
            sys.exit(1)
    
    def is_valid_move(self, dx, dy):
        """Check if the player's move is valid"""
        x, y = self.player_pos
        new_x, new_y = x + dx, y + dy
        
        # Check if the new position is within bounds
        if new_x < 0 or new_x >= len(self.level[0]) or new_y < 0 or new_y >= len(self.level):
            return False
        
        # Check what's at the new position
        new_cell = self.level[new_y][new_x]
        
        # If it's a wall, can't move
        if new_cell == WALL:
            return False
        
        # If it's a floor or target, can move
        if new_cell == FLOOR or new_cell == TARGET:
            return True
        
        # If it's a box or box on target, check if we can push it
        if new_cell == BOX or new_cell == BOX_ON_TARGET:
            # Calculate the position after the box
            box_x, box_y = new_x + dx, new_y + dy
            
            # Check if it's within bounds
            if box_x < 0 or box_x >= len(self.level[0]) or box_y < 0 or box_y >= len(self.level):
                return False
            
            # Check what's after the box
            after_box = self.level[box_y][box_x]
            
            # If it's a floor or target, can push
            if after_box == FLOOR or after_box == TARGET:
                return True
            
            # Otherwise, can't push
            return False
        
        # Otherwise, can't move
        return False
    
    def move_player(self, dx, dy):
        """Move the player in the given direction"""
        if self.is_completed:
            return
            
        if not self.is_valid_move(dx, dy):
            return
        
        x, y = self.player_pos
        new_x, new_y = x + dx, y + dy
        
        # Get current cells
        current_cell = self.level[y][x]
        new_cell = self.level[new_y][new_x]
        
        # Update the player position
        self.player_pos = (new_x, new_y)
        self.moves += 1
        
        # Handle box pushing
        if new_cell == BOX or new_cell == BOX_ON_TARGET:
            box_x, box_y = new_x + dx, new_y + dy
            box_cell = self.level[box_y][box_x]
            
            # Update box position
            if box_cell == FLOOR:
                self.level[box_y][box_x] = BOX
            elif box_cell == TARGET:
                self.level[box_y][box_x] = BOX_ON_TARGET
        
        # Update player position
        if current_cell == PLAYER:
            self.level[y][x] = FLOOR
        elif current_cell == PLAYER_ON_TARGET:
            self.level[y][x] = TARGET
            
        if new_cell == FLOOR or new_cell == BOX:
            self.level[new_y][new_x] = PLAYER
        elif new_cell == TARGET or new_cell == BOX_ON_TARGET:
            self.level[new_y][new_x] = PLAYER_ON_TARGET
        
        # Check if the level is completed
        self.check_completion()
    
    def check_completion(self):
        """Check if all boxes are on targets"""
        for row in self.level:
            for cell in row:
                if cell == BOX:  # If there's a box not on a target
                    return
        self.is_completed = True
    
    def draw(self):
        """Draw the game level"""
        self.screen.fill(BLACK)
        
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                
                # Draw the floor for all cells
                pygame.draw.rect(self.screen, FLOOR_COLOR, rect)
                
                # Draw the specific cell
                if cell == WALL:
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
                elif cell == PLAYER:
                    pygame.draw.rect(self.screen, FLOOR_COLOR, rect)
                    pygame.draw.circle(self.screen, PLAYER_COLOR, 
                                       (x * TILE_SIZE + TILE_SIZE // 2, 
                                        y * TILE_SIZE + TILE_SIZE // 2), 
                                       TILE_SIZE // 2 - 5)
                elif cell == TARGET:
                    pygame.draw.rect(self.screen, FLOOR_COLOR, rect)
                    pygame.draw.circle(self.screen, TARGET_COLOR, 
                                       (x * TILE_SIZE + TILE_SIZE // 2, 
                                        y * TILE_SIZE + TILE_SIZE // 2), 
                                       TILE_SIZE // 4)
                elif cell == BOX:
                    pygame.draw.rect(self.screen, FLOOR_COLOR, rect)
                    inner_rect = pygame.Rect(x * TILE_SIZE + 5, y * TILE_SIZE + 5, 
                                            TILE_SIZE - 10, TILE_SIZE - 10)
                    pygame.draw.rect(self.screen, BOX_COLOR, inner_rect)
                elif cell == BOX_ON_TARGET:
                    pygame.draw.rect(self.screen, FLOOR_COLOR, rect)
                    inner_rect = pygame.Rect(x * TILE_SIZE + 5, y * TILE_SIZE + 5, 
                                            TILE_SIZE - 10, TILE_SIZE - 10)
                    pygame.draw.rect(self.screen, BOX_ON_TARGET_COLOR, inner_rect)
                    pygame.draw.circle(self.screen, TARGET_COLOR, 
                                       (x * TILE_SIZE + TILE_SIZE // 2, 
                                        y * TILE_SIZE + TILE_SIZE // 2), 
                                       TILE_SIZE // 4)
                elif cell == PLAYER_ON_TARGET:
                    pygame.draw.rect(self.screen, FLOOR_COLOR, rect)
                    pygame.draw.circle(self.screen, TARGET_COLOR, 
                                       (x * TILE_SIZE + TILE_SIZE // 2, 
                                        y * TILE_SIZE + TILE_SIZE // 2), 
                                       TILE_SIZE // 4)
                    pygame.draw.circle(self.screen, PLAYER_COLOR, 
                                       (x * TILE_SIZE + TILE_SIZE // 2, 
                                        y * TILE_SIZE + TILE_SIZE // 2), 
                                       TILE_SIZE // 3)
        
        # Draw the number of moves
        font = pygame.font.SysFont(None, 24)
        moves_text = font.render(f"Moves: {self.moves}", True, WHITE)
        self.screen.blit(moves_text, (10, 10))
        
        # Draw level completion message
        if self.is_completed:
            font = pygame.font.SysFont(None, 48)
            complete_text = font.render("Level Complete!", True, WHITE)
            text_rect = complete_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(complete_text, text_rect)
            
            font = pygame.font.SysFont(None, 24)
            next_text = font.render("Press N for next level", True, WHITE)
            next_rect = next_text.get_rect(center=(self.width // 2, self.height // 2 + 40))
            self.screen.blit(next_text, next_rect)
    
    def next_level(self):
        """Load the next level"""
        if self.level_idx < len(LEVELS) - 1:
            self.level_idx += 1
            self.load_level(self.level_idx)
            self.moves = 0
            self.is_completed = False
            
            # Resize the screen for the new level
            self.width = len(self.level[0]) * TILE_SIZE
            self.height = len(self.level) * TILE_SIZE
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            print("You've completed all levels!")
    
    def reset_level(self):
        """Reset the current level"""
        self.load_level(self.level_idx)
        self.moves = 0
        self.is_completed = False
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.move_player(0, -1)
                    elif event.key == pygame.K_DOWN:
                        self.move_player(0, 1)
                    elif event.key == pygame.K_LEFT:
                        self.move_player(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.move_player(1, 0)
                    elif event.key == pygame.K_r:
                        self.reset_level()
                    elif event.key == pygame.K_n and self.is_completed:
                        self.next_level()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Draw the game
            self.draw()
            
            # Update the display
            pygame.display.flip()
            
            # Cap the FPS
            self.clock.tick(FPS)
        
        # Quit Pygame
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = SokobanGame()
    game.run() 