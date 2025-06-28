"""
Advanced game visualization for Snake AI with decision overlay
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
from typing import Dict, List, Tuple, Optional, Any
import colorsys
import time

from game.game_state import GameState
from utils.logger import setup_logger

class GameVisualizer:
    """Advanced game visualizer with AI decision overlay"""
    
    def __init__(self, width: int = 800, height: int = 600, grid_size: int = 20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cell_size = min(width, height) // grid_size
        self.logger = setup_logger('game_visualizer')
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Advanced Snake AI - Decision Visualization")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Colors
        self.colors = {
            'background': (15, 15, 25),
            'grid': (40, 40, 60),
            'snake_head': (100, 255, 100),
            'snake_body': (80, 200, 80),
            'food': (255, 100, 100),
            'text': (255, 255, 255),
            'ai_overlay': (255, 255, 100),
            'attention_high': (255, 0, 0),
            'attention_low': (0, 0, 255),
            'decision_path': (255, 165, 0),
            'safe_zone': (0, 255, 0),
            'danger_zone': (255, 0, 0),
            'exploration': (128, 0, 128)
        }
        
        # Visualization modes
        self.show_attention = True
        self.show_decision_path = True
        self.show_safety_zones = True
        self.show_exploration_map = False
        self.show_q_values = True
        
        # Animation
        self.animation_frame = 0
        self.decision_history = []
        
    def render_game_with_ai(self, game_state: GameState, ai_analysis: Dict) -> pygame.Surface:
        """Render game with AI decision overlay"""
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Calculate game board position (centered)
        board_width = self.grid_size * self.cell_size
        board_height = self.grid_size * self.cell_size
        board_x = (self.width - board_width) // 2
        board_y = 50  # Leave space for UI
        
        # Draw AI visualizations first (so they appear behind game elements)
        if ai_analysis:
            self._draw_ai_overlays(board_x, board_y, game_state, ai_analysis)
        
        # Draw grid
        self._draw_grid(board_x, board_y)
        
        # Draw game elements
        self._draw_snake(board_x, board_y, game_state.snake)
        self._draw_food(board_x, board_y, game_state.food)
        
        # Draw AI information panel
        self._draw_ai_info_panel(ai_analysis)
        
        # Draw game statistics
        self._draw_game_stats(game_state)
        
        # Update animation frame
        self.animation_frame += 1
        
        return self.screen
    
    def _draw_grid(self, offset_x: int, offset_y: int):
        """Draw game grid"""
        for x in range(self.grid_size + 1):
            start_pos = (offset_x + x * self.cell_size, offset_y)
            end_pos = (offset_x + x * self.cell_size, offset_y + self.grid_size * self.cell_size)
            pygame.draw.line(self.screen, self.colors['grid'], start_pos, end_pos, 1)
        
        for y in range(self.grid_size + 1):
            start_pos = (offset_x, offset_y + y * self.cell_size)
            end_pos = (offset_x + self.grid_size * self.cell_size, offset_y + y * self.cell_size)
            pygame.draw.line(self.screen, self.colors['grid'], start_pos, end_pos, 1)
    
    def _draw_snake(self, offset_x: int, offset_y: int, snake: List[Tuple[int, int]]):
        """Draw snake with enhanced visuals"""
        if not snake:
            return
        
        for i, (x, y) in enumerate(snake):
            cell_x = offset_x + x * self.cell_size
            cell_y = offset_y + y * self.cell_size
            
            if i == 0:  # Head
                # Draw head with gradient
                color = self.colors['snake_head']
                self._draw_gradient_rect(cell_x + 2, cell_y + 2, 
                                       self.cell_size - 4, self.cell_size - 4, 
                                       color, 0.8)
                
                # Draw direction indicator
                self._draw_snake_eyes(cell_x, cell_y, snake)
                
            else:  # Body
                # Body segments get darker towards tail
                fade_factor = 1.0 - (i / len(snake)) * 0.3
                color = tuple(int(c * fade_factor) for c in self.colors['snake_body'])
                
                pygame.draw.rect(self.screen, color,
                               (cell_x + 2, cell_y + 2, 
                                self.cell_size - 4, self.cell_size - 4))
                
                # Add segment connection lines
                if i < len(snake) - 1:
                    self._draw_segment_connection(offset_x, offset_y, snake[i], snake[i+1])
    
    def _draw_snake_eyes(self, cell_x: int, cell_y: int, snake: List[Tuple[int, int]]):
        """Draw snake eyes indicating direction"""
        if len(snake) < 2:
            return
        
        head_x, head_y = snake[0]
        next_x, next_y = snake[1]
        
        # Determine direction
        dx = head_x - next_x
        dy = head_y - next_y
        
        center_x = cell_x + self.cell_size // 2
        center_y = cell_y + self.cell_size // 2
        
        eye_size = 3
        eye_offset = 4
        
        if dx > 0:  # Moving right
            eye1_pos = (center_x + eye_offset, center_y - eye_offset)
            eye2_pos = (center_x + eye_offset, center_y + eye_offset)
        elif dx < 0:  # Moving left
            eye1_pos = (center_x - eye_offset, center_y - eye_offset)
            eye2_pos = (center_x - eye_offset, center_y + eye_offset)
        elif dy > 0:  # Moving down
            eye1_pos = (center_x - eye_offset, center_y + eye_offset)
            eye2_pos = (center_x + eye_offset, center_y + eye_offset)
        else:  # Moving up
            eye1_pos = (center_x - eye_offset, center_y - eye_offset)
            eye2_pos = (center_x + eye_offset, center_y - eye_offset)
        
        pygame.draw.circle(self.screen, (255, 255, 255), eye1_pos, eye_size)
        pygame.draw.circle(self.screen, (255, 255, 255), eye2_pos, eye_size)
        pygame.draw.circle(self.screen, (0, 0, 0), eye1_pos, eye_size - 1)
        pygame.draw.circle(self.screen, (0, 0, 0), eye2_pos, eye_size - 1)
    
    def _draw_food(self, offset_x: int, offset_y: int, food: Tuple[int, int]):
        """Draw food with pulsing animation"""
        x, y = food
        cell_x = offset_x + x * self.cell_size
        cell_y = offset_y + y * self.cell_size
        
        # Pulsing effect
        pulse = abs(np.sin(self.animation_frame * 0.1)) * 0.3 + 0.7
        size = int((self.cell_size - 6) * pulse)
        
        center_x = cell_x + self.cell_size // 2
        center_y = cell_y + self.cell_size // 2
        
        # Draw food with glow effect
        glow_color = tuple(int(c * 0.5) for c in self.colors['food'])
        pygame.draw.circle(self.screen, glow_color, (center_x, center_y), size + 4)
        pygame.draw.circle(self.screen, self.colors['food'], (center_x, center_y), size)
        
        # Add sparkle effect
        if self.animation_frame % 30 < 15:
            sparkle_positions = [
                (center_x - size//2, center_y - size//2),
                (center_x + size//2, center_y - size//2),
                (center_x, center_y + size//2)
            ]
            for pos in sparkle_positions:
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 2)
    
    def _draw_ai_overlays(self, offset_x: int, offset_y: int, 
                         game_state: GameState, ai_analysis: Dict):
        """Draw AI decision overlays"""
        
        # Draw attention heatmap
        if self.show_attention and 'attention_map' in ai_analysis:
            self._draw_attention_heatmap(offset_x, offset_y, ai_analysis['attention_map'])
        
        # Draw Q-value visualization
        if self.show_q_values and 'q_values' in ai_analysis:
            self._draw_q_value_arrows(offset_x, offset_y, game_state, ai_analysis['q_values'])
        
        # Draw safety zones
        if self.show_safety_zones:
            self._draw_safety_zones(offset_x, offset_y, game_state)
        
        # Draw decision path
        if self.show_decision_path and 'predicted_path' in ai_analysis:
            self._draw_decision_path(offset_x, offset_y, ai_analysis['predicted_path'])
        
        # Draw exploration map
        if self.show_exploration_map and 'exploration_map' in ai_analysis:
            self._draw_exploration_map(offset_x, offset_y, ai_analysis['exploration_map'])
    
    def _draw_attention_heatmap(self, offset_x: int, offset_y: int, attention_map: np.ndarray):
        """Draw attention heatmap overlay"""
        if attention_map.shape != (self.grid_size, self.grid_size):
            return
        
        # Create surface with alpha for transparency
        heatmap_surface = pygame.Surface((self.grid_size * self.cell_size, 
                                        self.grid_size * self.cell_size))
        heatmap_surface.set_alpha(100)
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                attention_value = attention_map[y, x]
                
                if attention_value > 0.1:  # Only show significant attention
                    # Color from blue (low) to red (high)
                    color_intensity = min(1.0, attention_value)
                    color = self._lerp_color(self.colors['attention_low'], 
                                           self.colors['attention_high'], 
                                           color_intensity)
                    
                    pygame.draw.rect(heatmap_surface, color,
                                   (x * self.cell_size, y * self.cell_size,
                                    self.cell_size, self.cell_size))
        
        self.screen.blit(heatmap_surface, (offset_x, offset_y))
    
    def _draw_q_value_arrows(self, offset_x: int, offset_y: int, 
                           game_state: GameState, q_values: List[float]):
        """Draw arrows indicating Q-values for each action"""
        if not game_state.snake or len(q_values) != 4:
            return
        
        head_x, head_y = game_state.snake[0]
        center_x = offset_x + head_x * self.cell_size + self.cell_size // 2
        center_y = offset_y + head_y * self.cell_size + self.cell_size // 2
        
        # Action directions: UP, RIGHT, DOWN, LEFT
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        
        max_q = max(q_values) if max(q_values) > 0 else 1
        
        for i, (dx, dy) in enumerate(directions):
            if i < len(q_values):
                q_value = q_values[i]
                normalized_q = q_value / max_q if max_q > 0 else 0
                
                # Arrow length and color based on Q-value
                arrow_length = int(abs(normalized_q) * self.cell_size * 0.8)
                
                if normalized_q > 0:
                    color = (0, 255, 0)  # Green for positive Q-values
                else:
                    color = (255, 0, 0)  # Red for negative Q-values
                
                # Arrow end position
                end_x = center_x + dx * arrow_length
                end_y = center_y + dy * arrow_length
                
                if arrow_length > 5:  # Only draw significant arrows
                    self._draw_arrow(center_x, center_y, end_x, end_y, color, 3)
                    
                    # Draw Q-value text
                    text = self.font_small.render(f"{q_value:.2f}", True, color)
                    text_x = end_x + dx * 15 - text.get_width() // 2
                    text_y = end_y + dy * 15 - text.get_height() // 2
                    self.screen.blit(text, (text_x, text_y))
    
    def _draw_safety_zones(self, offset_x: int, offset_y: int, game_state: GameState):
        """Draw safety zones around the snake"""
        if not game_state.snake:
            return
        
        head_x, head_y = game_state.snake[0]
        
        # Check safety in 3x3 grid around head
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                check_x = head_x + dx
                check_y = head_y + dy
                
                if (0 <= check_x < self.grid_size and 
                    0 <= check_y < self.grid_size and
                    (check_x, check_y) not in game_state.snake):
                    
                    # Calculate safety score
                    safety_score = self._calculate_position_safety(
                        game_state, check_x, check_y
                    )
                    
                    if safety_score > 0.5:
                        color = (*self.colors['safe_zone'], 50)
                    elif safety_score < -0.5:
                        color = (*self.colors['danger_zone'], 50)
                    else:
                        continue
                    
                    # Draw safety indicator
                    cell_x = offset_x + check_x * self.cell_size
                    cell_y = offset_y + check_y * self.cell_size
                    
                    safety_surface = pygame.Surface((self.cell_size, self.cell_size))
                    safety_surface.set_alpha(50)
                    safety_surface.fill(color[:3])
                    self.screen.blit(safety_surface, (cell_x, cell_y))
    
    def _calculate_position_safety(self, game_state: GameState, x: int, y: int) -> float:
        """Calculate safety score for a position"""
        safety_score = 0
        
        # Check surrounding positions
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                check_x, check_y = x + dx, y + dy
                
                # Wall proximity penalty
                if (check_x < 0 or check_x >= self.grid_size or 
                    check_y < 0 or check_y >= self.grid_size):
                    safety_score -= 1
                
                # Snake body proximity penalty
                elif (check_x, check_y) in game_state.snake:
                    safety_score -= 2
                
                # Free space bonus
                else:
                    safety_score += 0.5
        
        return safety_score / 9  # Normalize
    
    def _draw_decision_path(self, offset_x: int, offset_y: int, predicted_path: List[Tuple[int, int]]):
        """Draw predicted decision path"""
        if len(predicted_path) < 2:
            return
        
        # Draw path as connected lines
        for i in range(len(predicted_path) - 1):
            start_x, start_y = predicted_path[i]
            end_x, end_y = predicted_path[i + 1]
            
            start_pixel_x = offset_x + start_x * self.cell_size + self.cell_size // 2
            start_pixel_y = offset_y + start_y * self.cell_size + self.cell_size // 2
            end_pixel_x = offset_x + end_x * self.cell_size + self.cell_size // 2
            end_pixel_y = offset_y + end_y * self.cell_size + self.cell_size // 2
            
            # Fade path over distance
            alpha = max(50, 255 - i * 30)
            color = (*self.colors['decision_path'], alpha)
            
            self._draw_thick_line(start_pixel_x, start_pixel_y, 
                                end_pixel_x, end_pixel_y, color, 4)
    
    def _draw_ai_info_panel(self, ai_analysis: Dict):
        """Draw AI information panel"""
        panel_y = 10
        
        # AI model information
        if 'model_type' in ai_analysis:
            text = self.font_medium.render(f"AI Model: {ai_analysis['model_type']}", 
                                         True, self.colors['text'])
            self.screen.blit(text, (10, panel_y))
            panel_y += 25
        
        # Action probabilities
        if 'action_probabilities' in ai_analysis:
            probs = ai_analysis['action_probabilities']
            actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
            
            text = self.font_medium.render("Action Probabilities:", 
                                         True, self.colors['text'])
            self.screen.blit(text, (10, panel_y))
            panel_y += 20
            
            for i, (action, prob) in enumerate(zip(actions, probs)):
                color = self._get_probability_color(prob)
                bar_width = int(prob * 100)
                
                # Draw probability bar
                pygame.draw.rect(self.screen, color,
                               (100 + i * 120, panel_y, bar_width, 15))
                pygame.draw.rect(self.screen, self.colors['grid'],
                               (100 + i * 120, panel_y, 100, 15), 1)
                
                # Action label
                label = self.font_small.render(f"{action}: {prob:.2f}", 
                                             True, self.colors['text'])
                self.screen.blit(label, (100 + i * 120, panel_y + 18))
        
        # AI confidence
        if 'confidence' in ai_analysis:
            confidence = ai_analysis['confidence']
            text = self.font_medium.render(f"AI Confidence: {confidence:.2%}", 
                                         True, self.colors['text'])
            self.screen.blit(text, (self.width - 200, 10))
        
        # Decision reasoning
        if 'reasoning' in ai_analysis:
            reasoning = ai_analysis['reasoning']
            y_pos = self.height - 100
            
            text = self.font_medium.render("AI Reasoning:", True, self.colors['text'])
            self.screen.blit(text, (10, y_pos))
            y_pos += 25
            
            # Wrap text for reasoning
            words = reasoning.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                if self.font_small.size(test_line)[0] > self.width - 20:
                    if line:
                        text = self.font_small.render(line, True, self.colors['text'])
                        self.screen.blit(text, (10, y_pos))
                        y_pos += 20
                    line = word + " "
                else:
                    line = test_line
            
            if line:
                text = self.font_small.render(line, True, self.colors['text'])
                self.screen.blit(text, (10, y_pos))
    
    def _draw_game_stats(self, game_state: GameState):
        """Draw game statistics"""
        stats_x = self.width - 150
        stats_y = 50
        
        stats = [
            f"Score: {game_state.score}",
            f"Length: {len(game_state.snake)}",
            f"Steps: {game_state.steps}",
            f"Efficiency: {game_state.score / max(1, game_state.steps):.3f}"
        ]
        
        for stat in stats:
            text = self.font_medium.render(stat, True, self.colors['text'])
            self.screen.blit(text, (stats_x, stats_y))
            stats_y += 25
    
    # Helper methods
    def _draw_gradient_rect(self, x: int, y: int, width: int, height: int, 
                           color: Tuple[int, int, int], alpha: float):
        """Draw rectangle with gradient effect"""
        surface = pygame.Surface((width, height))
        surface.set_alpha(int(alpha * 255))
        
        for i in range(height):
            fade = 1.0 - (i / height) * 0.3
            fade_color = tuple(int(c * fade) for c in color)
            pygame.draw.line(surface, fade_color, (0, i), (width, i))
        
        self.screen.blit(surface, (x, y))
    
    def _draw_arrow(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                   color: Tuple[int, int, int], thickness: int):
        """Draw arrow from start to end"""
        # Draw line
        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), thickness)
        
        # Calculate arrow head
        arrow_length = 8
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Normalize direction
            dx /= length
            dy /= length
            
            # Arrow head points
            head_x1 = end_x - arrow_length * dx + arrow_length * 0.5 * dy
            head_y1 = end_y - arrow_length * dy - arrow_length * 0.5 * dx
            head_x2 = end_x - arrow_length * dx - arrow_length * 0.5 * dy
            head_y2 = end_y - arrow_length * dy + arrow_length * 0.5 * dx
            
            # Draw arrow head
            pygame.draw.polygon(self.screen, color, 
                              [(end_x, end_y), (head_x1, head_y1), (head_x2, head_y2)])
    
    def _draw_thick_line(self, start_x: int, start_y: int, end_x: int, end_y: int,
                        color: Tuple[int, int, int, int], thickness: int):
        """Draw thick line with alpha"""
        # Create surface for alpha blending
        line_surface = pygame.Surface((abs(end_x - start_x) + thickness, 
                                     abs(end_y - start_y) + thickness))
        line_surface.set_alpha(color[3] if len(color) > 3 else 255)
        
        # Adjust coordinates for surface
        local_start_x = thickness // 2
        local_start_y = thickness // 2
        local_end_x = local_start_x + (end_x - start_x)
        local_end_y = local_start_y + (end_y - start_y)
        
        pygame.draw.line(line_surface, color[:3], 
                        (local_start_x, local_start_y), 
                        (local_end_x, local_end_y), thickness)
        
        # Blit to main screen
        blit_x = min(start_x, end_x) - thickness // 2
        blit_y = min(start_y, end_y) - thickness // 2
        self.screen.blit(line_surface, (blit_x, blit_y))
    
    def _lerp_color(self, color1: Tuple[int, int, int], 
                   color2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        """Linear interpolation between two colors"""
        return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))
    
    def _get_probability_color(self, probability: float) -> Tuple[int, int, int]:
        """Get color based on probability value"""
        # Green for high probability, red for low
        if probability > 0.7:
            return (0, 255, 0)
        elif probability > 0.4:
            return (255, 255, 0)
        else:
            return (255, 0, 0)
    
    def _draw_segment_connection(self, offset_x: int, offset_y: int,
                               segment1: Tuple[int, int], segment2: Tuple[int, int]):
        """Draw connection between snake segments"""
        x1, y1 = segment1
        x2, y2 = segment2
        
        center1_x = offset_x + x1 * self.cell_size + self.cell_size // 2
        center1_y = offset_y + y1 * self.cell_size + self.cell_size // 2
        center2_x = offset_x + x2 * self.cell_size + self.cell_size // 2
        center2_y = offset_y + y2 * self.cell_size + self.cell_size // 2
        
        pygame.draw.line(self.screen, self.colors['snake_body'],
                        (center1_x, center1_y), (center2_x, center2_y), 3)
    
    def _draw_exploration_map(self, offset_x: int, offset_y: int, exploration_map: np.ndarray):
        """Draw exploration heatmap"""
        if exploration_map.shape != (self.grid_size, self.grid_size):
            return
        
        max_visits = np.max(exploration_map) if np.max(exploration_map) > 0 else 1
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                visits = exploration_map[y, x]
                if visits > 0:
                    # Color intensity based on visit count
                    intensity = min(1.0, visits / max_visits)
                    color = tuple(int(c * intensity) for c in self.colors['exploration'])
                    
                    pygame.draw.rect(self.screen, color,
                                   (offset_x + x * self.cell_size + 1,
                                    offset_y + y * self.cell_size + 1,
                                    self.cell_size - 2, self.cell_size - 2))
    
    def toggle_visualization_mode(self, mode: str):
        """Toggle visualization modes"""
        if mode == 'attention':
            self.show_attention = not self.show_attention
        elif mode == 'decision_path':
            self.show_decision_path = not self.show_decision_path
        elif mode == 'safety_zones':
            self.show_safety_zones = not self.show_safety_zones
        elif mode == 'exploration':
            self.show_exploration_map = not self.show_exploration_map
        elif mode == 'q_values':
            self.show_q_values = not self.show_q_values
    
    def save_frame(self, filename: str):
        """Save current frame as image"""
        pygame.image.save(self.screen, filename)
    
    def create_video_frame(self) -> np.ndarray:
        """Create video frame as numpy array"""
        # Convert pygame surface to numpy array
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))  # Correct orientation
        return frame
    
    def cleanup(self):
        """Cleanup resources"""
        pygame.quit()

class TrainingVisualizer:
    """Visualizer for training progress and metrics"""
    
    def __init__(self):
        self.logger = setup_logger('training_visualizer')
        plt.style.use('dark_background')
    
    def plot_training_metrics(self, metrics: Dict) -> np.ndarray:
        """Plot training metrics and return as image array"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Snake AI Training Progress', fontsize=16, color='white')
        
        # Score progression
        if 'scores' in metrics and metrics['scores']:
            axes[0, 0].plot(metrics['scores'], color='lime', linewidth=2)
            axes[0, 0].set_title('Score Progression', color='white')
            axes[0, 0].set_xlabel('Episode', color='white')
            axes[0, 0].set_ylabel('Score', color='white')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if 'episode_lengths' in metrics and metrics['episode_lengths']:
            axes[0, 1].plot(metrics['episode_lengths'], color='cyan', linewidth=2)
            axes[0, 1].set_title('Episode Lengths', color='white')
            axes[0, 1].set_xlabel('Episode', color='white')
            axes[0, 1].set_ylabel('Steps', color='white')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Loss progression
        if 'losses' in metrics and metrics['losses']:
            axes[1, 0].plot(metrics['losses'], color='orange', linewidth=2)
            axes[1, 0].set_title('Training Loss', color='white')
            axes[1, 0].set_xlabel('Training Step', color='white')
            axes[1, 0].set_ylabel('Loss', color='white')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Q-values
        if 'q_values' in metrics and metrics['q_values']:
            axes[1, 1].plot(metrics['q_values'], color='yellow', linewidth=2)
            axes[1, 1].set_title('Average Q-Values', color='white')
            axes[1, 1].set_xlabel('Episode', color='white')
            axes[1, 1].set_ylabel('Q-Value', color='white')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Convert to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        plt.close(fig)
        
        # Convert to numpy array
        image_array = np.frombuffer(raw_data, dtype=np.uint8)
        image_array = image_array.reshape(size[1], size[0], 3)
        
        return image_array
    
    def plot_curriculum_progress(self, curriculum_stats: Dict) -> np.ndarray:
        """Plot curriculum learning progress"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Curriculum Learning Progress', fontsize=16, color='white')
        
        # Current stage progress
        current_stage = curriculum_stats.get('current_stage', {})
        if current_stage:
            progress = current_stage.get('progress', 0)
            success_rate = current_stage.get('success_rate', 0)
            
            # Progress bar
            axes[0].barh(['Progress'], [progress], color='lime', alpha=0.7)
            axes[0].barh(['Success Rate'], [success_rate], color='cyan', alpha=0.7)
            axes[0].set_xlim(0, 1)
            axes[0].set_title(f"Stage: {current_stage.get('name', 'Unknown')}", color='white')
            axes[0].grid(True, alpha=0.3)
        
        # Stage history
        progression_history = curriculum_stats.get('progression_history', [])
        if progression_history:
            stage_names = [stage['stage_name'] for stage in progression_history]
            success_rates = [stage['success_rate'] for stage in progression_history]
            
            axes[1].bar(range(len(stage_names)), success_rates, color='orange', alpha=0.7)
            axes[1].set_xticks(range(len(stage_names)))
            axes[1].set_xticklabels(stage_names, rotation=45, ha='right')
            axes[1].set_title('Stage Completion Success Rates', color='white')
            axes[1].set_ylabel('Success Rate', color='white')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        plt.close(fig)
        
        image_array = np.frombuffer(raw_data, dtype=np.uint8)
        image_array = image_array.reshape(size[1], size[0], 3)
        
        return image_array

