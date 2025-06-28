"""
Enhanced Snake Game implementation with AI integration
"""

import pygame
import numpy as np
from typing import Tuple, Optional, List
import time

from game.game_state import GameState, Direction
from config.settings import Config

class SnakeGameRenderer:
    """Visual renderer for the Snake game"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cell_size = config.GAME_WIDTH // config.GRID_SIZE
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((config.GAME_WIDTH, config.GAME_HEIGHT + 100))
        pygame.display.set_caption("Advanced Snake AI")
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.colors = {
            'background': (20, 20, 20),
            'snake_head': (0, 255, 0),
            'snake_body': (0, 200, 0),
            'food': (255, 0, 0),
            'grid': (40, 40, 40),
            'text': (255, 255, 255),
            'ai_overlay': (255, 255, 0, 100)
        }
    
    def render(self, game_state: GameState, ai_info: Optional[dict] = None):
        """Render the game state"""
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        self._draw_grid()
        
        # Draw snake
        self._draw_snake(game_state.snake)
        
        # Draw food
        self._draw_food(game_state.food)
        
        # Draw AI overlay if provided
        if ai_info:
            self._draw_ai_overlay(ai_info)
        
        # Draw game info
        self._draw_game_info(game_state, ai_info)
        
        pygame.display.flip()
    
    def _draw_grid(self):
        """Draw grid lines"""
        for x in range(0, self.config.GAME_WIDTH, self.cell_size):
            pygame.draw.line(self.screen, self.colors['grid'], 
                           (x, 0), (x, self.config.GAME_HEIGHT), 1)
        
        for y in range(0, self.config.GAME_HEIGHT, self.cell_size):
            pygame.draw.line(self.screen, self.colors['grid'], 
                           (0, y), (self.config.GAME_WIDTH, y), 1)
    
    def _draw_snake(self, snake: List[Tuple[int, int]]):
        """Draw the snake"""
        for i, (x, y) in enumerate(snake):
            color = self.colors['snake_head'] if i == 0 else self.colors['snake_body']
            rect = pygame.Rect(
                x * self.cell_size + 1,
                y * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.screen, color, rect)
            
            # Add direction indicator for head
            if i == 0 and len(snake) > 1:
                next_x, next_y = snake[1]
                center_x = x * self.cell_size + self.cell_size // 2
                center_y = y * self.cell_size + self.cell_size // 2
                
                # Draw eye-like indicators
                eye_size = 3
                if x > next_x:  # Moving right
                    pygame.draw.circle(self.screen, (255, 255, 255), 
                                     (center_x + 5, center_y - 3), eye_size)
                    pygame.draw.circle(self.screen, (255, 255, 255), 
                                     (center_x + 5, center_y + 3), eye_size)
                elif x < next_x:  # Moving left
                    pygame.draw.circle(self.screen, (255, 255, 255), 
                                     (center_x - 5, center_y - 3), eye_size)
                    pygame.draw.circle(self.screen, (255, 255, 255), 
                                     (center_x - 5, center_y + 3), eye_size)
                elif y > next_y:  # Moving down
                    pygame.draw.circle(self.screen, (255, 255, 255), 
                                     (center_x - 3, center_y + 5), eye_size)
                    pygame.draw.circle(self.screen, (255, 255, 255), 
                                     (center_x + 3, center_y + 5), eye_size)
                elif y < next_y:  # Moving up
                    pygame.draw.circle(self.screen, (255, 255, 255), 
                                     (center_x - 3, center_y - 5), eye_size)
                    pygame.draw.circle(self.screen, (255, 255, 255), 
                                     (center_x + 3, center_y - 5), eye_size)
    
    def _draw_food(self, food: Tuple[int, int]):
        """Draw the food"""
        x, y = food
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, self.colors['food'], 
                         (center_x, center_y), self.cell_size // 2 - 2)
    
    def _draw_ai_overlay(self, ai_info: dict):
        """Draw AI decision overlay"""
        if 'action_probabilities' in ai_info:
            probs = ai_info['action_probabilities']
            actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
            
            # Draw action probabilities as bars
            bar_width = 100
            bar_height = 20
            start_y = self.config.GAME_HEIGHT + 10
            
            for i, (action, prob) in enumerate(zip(actions, probs)):
                x = i * (bar_width + 10) + 10
                
                # Background bar
                pygame.draw.rect(self.screen, (60, 60, 60), 
                               (x, start_y, bar_width, bar_height))
                
                # Probability bar
                filled_width = int(bar_width * prob)
                color_intensity = int(255 * prob)
                color = (color_intensity, 255 - color_intensity, 0)
                pygame.draw.rect(self.screen, color, 
                               (x, start_y, filled_width, bar_height))
                
                # Text
                text = self.small_font.render(f"{action}: {prob:.2f}", 
                                            True, self.colors['text'])
                self.screen.blit(text, (x, start_y + bar_height + 5))
    
    def _draw_game_info(self, game_state: GameState, ai_info: Optional[dict]):
        """Draw game information"""
        y_offset = self.config.GAME_HEIGHT + 50
        
        # Basic game info
        score_text = self.font.render(f"Score: {game_state.score}", 
                                    True, self.colors['text'])
        self.screen.blit(score_text, (10, y_offset))
        
        steps_text = self.font.render(f"Steps: {game_state.steps}", 
                                    True, self.colors['text'])
        self.screen.blit(steps_text, (150, y_offset))
        
        # AI info
        if ai_info:
            if 'episode' in ai_info:
                episode_text = self.font.render(f"Episode: {ai_info['episode']}", 
                                              True, self.colors['text'])
                self.screen.blit(episode_text, (300, y_offset))
            
            if 'epsilon' in ai_info:
                epsilon_text = self.font.render(f"ε: {ai_info['epsilon']:.3f}", 
                                              True, self.colors['text'])
                self.screen.blit(epsilon_text, (450, y_offset))

class SnakeGame:
    """Main Snake game class with AI integration"""
    
    def __init__(self, config: Config, headless: bool = False):
        self.config = config
        self.headless = headless
        self.game_state = GameState(config.GRID_SIZE, config.GRID_SIZE)
        
        if not headless:
            self.renderer = SnakeGameRenderer(config)
        
        self.clock = pygame.time.Clock() if not headless else None
    
    def reset(self) -> np.ndarray:
        """Reset game and return initial state"""
        self.game_state.reset()
        return self.get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one game step"""
        reward, done = self.game_state.step(action)
        
        # Additional reward shaping
        if not done:
            reward += self._calculate_shaped_reward()
        
        info = {
            'score': self.game_state.score,
            'steps': self.game_state.steps,
            'food_eaten': self.game_state.food_eaten
        }
        
        return self.get_state(), reward, done, info
    
    def _calculate_shaped_reward(self) -> float:
        """Calculate additional shaped rewards"""
        if not self.game_state.snake:
            return 0
        
        head_x, head_y = self.game_state.snake[0]
        food_x, food_y = self.game_state.food
        
        # Distance-based reward
        distance = abs(head_x - food_x) + abs(head_y - food_y)
        max_distance = self.config.GRID_SIZE * 2
        distance_reward = (max_distance - distance) / max_distance * 0.1
        
        # Survival bonus
        survival_bonus = 0.01
        
        # Penalty for getting too close to walls or self
        danger_penalty = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_x, next_y = head_x + dx, head_y + dy
            if (next_x < 0 or next_x >= self.config.GRID_SIZE or 
                next_y < 0 or next_y >= self.config.GRID_SIZE or
                (next_x, next_y) in self.game_state.snake):
                danger_penalty -= 0.05
        
        return distance_reward + survival_bonus + danger_penalty
    
    def get_state(self) -> np.ndarray:
        """Get current game state for AI"""
        return self.game_state.get_state_array()
    
    def get_vision_state(self) -> np.ndarray:
        """Get vision-based state"""
        return self.game_state.get_vision_state()
    
    def get_compact_state(self) -> np.ndarray:
        """Get compact state representation"""
        return self.game_state.get_compact_state()
    
    def render(self, ai_info: Optional[dict] = None):
        """Render the game"""
        if not self.headless:
            self.renderer.render(self.game_state, ai_info)
            if self.clock:
                self.clock.tick(self.config.FPS)
    
    def handle_events(self) -> bool:
        """Handle pygame events"""
        if self.headless:
            return True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True
    
    def close(self):
        """Close the game"""
        if not self.headless:
            pygame.quit()
