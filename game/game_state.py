"""
Game state representation and utilities for Snake AI
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from enum import Enum
import cv2

class Direction(Enum):
    """Snake movement directions"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class GameState:
    """Advanced game state representation for AI"""
    
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.reset()
    
    def reset(self):
        """Reset game state to initial conditions"""
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = Direction.RIGHT
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.food_eaten = False
    
    def _place_food(self) -> Tuple[int, int]:
        """Place food randomly, avoiding snake body"""
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if (x, y) not in self.snake:
                return (x, y)
    
    def step(self, action: int) -> Tuple[float, bool]:
        """
        Execute one game step
        
        Args:
            action: Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            reward: Reward for the action
            done: Whether game is over
        """
        
        if self.game_over:
            return 0, True
        
        # Update direction based on action
        new_direction = Direction(action)
        
        # Prevent 180-degree turns
        opposite_directions = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        
        if new_direction != opposite_directions.get(self.direction):
            self.direction = new_direction
        
        # Move snake head
        head_x, head_y = self.snake[0]
        
        if self.direction == Direction.UP:
            head_y -= 1
        elif self.direction == Direction.DOWN:
            head_y += 1
        elif self.direction == Direction.LEFT:
            head_x -= 1
        elif self.direction == Direction.RIGHT:
            head_x += 1
        
        new_head = (head_x, head_y)
        
        # Check wall collision
        if (head_x < 0 or head_x >= self.width or 
            head_y < 0 or head_y >= self.height):
            self.game_over = True
            return -10, True  # Wall collision penalty
        
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return -10, True  # Self collision penalty
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check food collision
        self.food_eaten = False
        if new_head == self.food:
            self.score += 1
            self.food_eaten = True
            self.food = self._place_food()
            reward = 10  # Food reward
        else:
            self.snake.pop()  # Remove tail if no food eaten
            reward = 0
        
        self.steps += 1
        
        # Step penalty to encourage efficiency
        reward -= 0.01
        
        return reward, False
    
    def get_state_array(self) -> np.ndarray:
        """Get state as numpy array for neural network"""
        state = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Channel 0: Snake body
        for x, y in self.snake[1:]:
            state[y, x, 0] = 1.0
        
        # Channel 1: Snake head
        if self.snake:
            head_x, head_y = self.snake[0]
            state[head_y, head_x, 1] = 1.0
        
        # Channel 2: Food
        food_x, food_y = self.food
        state[food_y, food_x, 2] = 1.0
        
        return state
    
    def get_vision_state(self, vision_distance: int = 8) -> np.ndarray:
        """Get vision-based state representation"""
        if not self.snake:
            return np.zeros((vision_distance * 8 + 4,), dtype=np.float32)
        
        head_x, head_y = self.snake[0]
        vision = []
        
        # 8 directions for vision
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Up-left, Up, Up-right
            (0, -1),           (0, 1),   # Left, Right
            (1, -1),  (1, 0),  (1, 1)    # Down-left, Down, Down-right
        ]
        
        for dx, dy in directions:
            # Look in each direction
            for distance in range(1, vision_distance + 1):
                x = head_x + dx * distance
                y = head_y + dy * distance
                
                # Wall detection
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    vision.extend([1.0, 0.0, 0.0])  # Wall, no body, no food
                    break
                
                # Body detection
                body_detected = 1.0 if (x, y) in self.snake[1:] else 0.0
                
                # Food detection
                food_detected = 1.0 if (x, y) == self.food else 0.0
                
                vision.extend([0.0, body_detected, food_detected])
                
                # Stop if we hit something
                if body_detected or food_detected:
                    break
            else:
                # If we didn't break, add empty vision
                vision.extend([0.0, 0.0, 0.0])
        
        # Add direction information
        direction_vector = [0.0] * 4
        direction_vector[self.direction.value] = 1.0
        vision.extend(direction_vector)
        
        return np.array(vision, dtype=np.float32)
    
    def get_compact_state(self) -> np.ndarray:
        """Get compact state representation"""
        if not self.snake:
            return np.zeros(11, dtype=np.float32)
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Normalized positions
        head_x_norm = head_x / self.width
        head_y_norm = head_y / self.height
        food_x_norm = food_x / self.width
        food_y_norm = food_y / self.height
        
        # Distance to food
        food_distance = np.sqrt((head_x - food_x)**2 + (head_y - food_y)**2)
        food_distance_norm = food_distance / np.sqrt(self.width**2 + self.height**2)
        
        # Direction to food
        food_dx = (food_x - head_x) / self.width
        food_dy = (food_y - head_y) / self.height
        
        # Danger detection (next step collision)
        dangers = []
        for action in range(4):
            temp_state = GameState(self.width, self.height)
            temp_state.snake = self.snake.copy()
            temp_state.direction = self.direction
            temp_state.food = self.food
            temp_state.game_over = self.game_over
            
            reward, done = temp_state.step(action)
            dangers.append(1.0 if done else 0.0)
        
        return np.array([
            head_x_norm, head_y_norm,
            food_x_norm, food_y_norm,
            food_distance_norm,
            food_dx, food_dy,
            *dangers
        ], dtype=np.float32)
    
    def get_enhanced_features(self) -> Dict[str, Any]:
        """Get enhanced features for advanced AI"""
        if not self.snake:
            return {}
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        features = {
            'snake_length': len(self.snake),
            'food_distance': np.sqrt((head_x - food_x)**2 + (head_y - food_y)**2),
            'steps_since_food': self.steps,
            'board_coverage': len(self.snake) / (self.width * self.height),
            'head_to_center_distance': np.sqrt((head_x - self.width//2)**2 + (head_y - self.height//2)**2),
            'food_to_center_distance': np.sqrt((food_x - self.width//2)**2 + (food_y - self.height//2)**2),
            'direction_value': self.direction.value,
            'score': self.score
        }
        
        return features
