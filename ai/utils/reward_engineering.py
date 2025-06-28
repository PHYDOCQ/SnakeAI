"""
Advanced reward engineering for Snake AI training
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import math

from game.game_state import GameState, Direction
from utils.logger import setup_logger

class RewardType(Enum):
    """Types of rewards in the system"""
    SURVIVAL = "survival"
    FOOD = "food"
    DISTANCE = "distance"
    EXPLORATION = "exploration"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    STRATEGIC = "strategic"
    PROGRESS = "progress"

class RewardShaper:
    """Advanced reward shaping for Snake AI"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = setup_logger('reward_shaper')
        
        # Reward weights
        self.reward_weights = {
            RewardType.SURVIVAL: 0.01,
            RewardType.FOOD: 10.0,
            RewardType.DISTANCE: 0.1,
            RewardType.EXPLORATION: 0.05,
            RewardType.EFFICIENCY: 0.02,
            RewardType.SAFETY: 0.15,
            RewardType.STRATEGIC: 0.08,
            RewardType.PROGRESS: 0.03
        }
        
        # Reward history for adaptive shaping
        self.reward_history = []
        self.state_visit_counts = {}
        self.path_history = []
        
        # Dynamic parameters
        self.exploration_decay = 0.999
        self.safety_sensitivity = 1.0
        self.efficiency_threshold = 0.5
        
    def shape_reward(self, base_reward: float, game_state: GameState, 
                    info: Dict, curriculum_config: Optional[Dict] = None) -> float:
        """
        Shape reward based on multiple factors
        
        Args:
            base_reward: Original reward from environment
            game_state: Current game state
            info: Additional information from environment
            curriculum_config: Current curriculum configuration
            
        Returns:
            Shaped reward value
        """
        
        shaped_rewards = {}
        
        # Base reward (food collection, death penalty)
        shaped_rewards['base'] = base_reward
        
        # Survival reward
        shaped_rewards[RewardType.SURVIVAL] = self._calculate_survival_reward(game_state)
        
        # Distance-based reward
        shaped_rewards[RewardType.DISTANCE] = self._calculate_distance_reward(game_state)
        
        # Exploration reward
        shaped_rewards[RewardType.EXPLORATION] = self._calculate_exploration_reward(game_state)
        
        # Efficiency reward
        shaped_rewards[RewardType.EFFICIENCY] = self._calculate_efficiency_reward(game_state, info)
        
        # Safety reward
        shaped_rewards[RewardType.SAFETY] = self._calculate_safety_reward(game_state)
        
        # Strategic reward
        shaped_rewards[RewardType.STRATEGIC] = self._calculate_strategic_reward(game_state)
        
        # Progress reward
        shaped_rewards[RewardType.PROGRESS] = self._calculate_progress_reward(game_state)
        
        # Apply curriculum multipliers
        if curriculum_config:
            curriculum_multiplier = curriculum_config.get('reward_multiplier', 1.0)
            for reward_type in shaped_rewards:
                if reward_type != 'base':
                    shaped_rewards[reward_type] *= curriculum_multiplier
        
        # Combine rewards
        total_reward = shaped_rewards['base']
        
        for reward_type, reward_value in shaped_rewards.items():
            if reward_type != 'base' and isinstance(reward_type, RewardType):
                weight = self.reward_weights.get(reward_type, 0.0)
                total_reward += weight * reward_value
        
        # Store for analysis
        self.reward_history.append({
            'total': total_reward,
            'components': shaped_rewards,
            'game_state': {
                'score': game_state.score,
                'steps': game_state.steps,
                'snake_length': len(game_state.snake)
            }
        })
        
        # Keep only recent history
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        
        return total_reward
    
    def _calculate_survival_reward(self, game_state: GameState) -> float:
        """Calculate reward for staying alive"""
        if game_state.game_over:
            return -1.0
        
        # Small positive reward for each step survived
        base_survival = 1.0
        
        # Bonus for longer survival with larger snake
        length_bonus = math.log(1 + len(game_state.snake)) * 0.1
        
        return base_survival + length_bonus
    
    def _calculate_distance_reward(self, game_state: GameState) -> float:
        """Calculate reward based on distance to food"""
        if not game_state.snake:
            return 0
        
        head_x, head_y = game_state.snake[0]
        food_x, food_y = game_state.food
        
        # Manhattan distance
        distance = abs(head_x - food_x) + abs(head_y - food_y)
        max_distance = game_state.width + game_state.height
        
        # Reward for getting closer to food
        normalized_distance = distance / max_distance
        distance_reward = (1.0 - normalized_distance) * 2.0 - 1.0  # Range [-1, 1]
        
        # Bonus for being very close
        if distance <= 2:
            distance_reward += 0.5
        
        return distance_reward
    
    def _calculate_exploration_reward(self, game_state: GameState) -> float:
        """Calculate reward for exploring new areas"""
        if not game_state.snake:
            return 0
        
        head_pos = game_state.snake[0]
        
        # Count visits to this position
        visit_count = self.state_visit_counts.get(head_pos, 0)
        self.state_visit_counts[head_pos] = visit_count + 1
        
        # Reward decreases with repeated visits
        exploration_reward = 1.0 / (1.0 + visit_count)
        
        # Apply exploration decay
        exploration_reward *= self.exploration_decay ** game_state.steps
        
        return exploration_reward
    
    def _calculate_efficiency_reward(self, game_state: GameState, info: Dict) -> float:
        """Calculate reward for efficient food collection"""
        if game_state.steps == 0:
            return 0
        
        # Efficiency = score / steps
        efficiency = game_state.score / game_state.steps
        
        # Reward high efficiency
        if efficiency > self.efficiency_threshold:
            return (efficiency - self.efficiency_threshold) * 5.0
        else:
            return (efficiency - self.efficiency_threshold) * 2.0
    
    def _calculate_safety_reward(self, game_state: GameState) -> float:
        """Calculate reward for safe behavior"""
        if not game_state.snake:
            return 0
        
        head_x, head_y = game_state.snake[0]
        safety_score = 0
        
        # Check danger in all directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        safe_directions = 0
        
        for dx, dy in directions:
            next_x, next_y = head_x + dx, head_y + dy
            
            # Check if position is safe
            is_safe = (
                0 <= next_x < game_state.width and
                0 <= next_y < game_state.height and
                (next_x, next_y) not in game_state.snake
            )
            
            if is_safe:
                safe_directions += 1
        
        # Reward having multiple safe escape routes
        safety_score = safe_directions / 4.0
        
        # Penalty for being in dangerous situations
        if safe_directions <= 1:
            safety_score -= 0.5
        
        # Additional check for potential traps
        trap_penalty = self._check_for_traps(game_state)
        safety_score -= trap_penalty
        
        return safety_score * self.safety_sensitivity
    
    def _check_for_traps(self, game_state: GameState) -> float:
        """Check if snake is heading into a potential trap"""
        if len(game_state.snake) < 4:
            return 0  # Short snake unlikely to trap itself
        
        head_x, head_y = game_state.snake[0]
        
        # Use flood fill to check available space
        available_space = self._flood_fill_available_space(
            game_state, head_x, head_y
        )
        
        # If available space is less than snake length, it's a potential trap
        snake_length = len(game_state.snake)
        
        if available_space < snake_length:
            return 1.0  # High trap penalty
        elif available_space < snake_length * 1.5:
            return 0.5  # Medium trap penalty
        else:
            return 0  # No trap
    
    def _flood_fill_available_space(self, game_state: GameState, 
                                  start_x: int, start_y: int) -> int:
        """Count available space using flood fill algorithm"""
        visited = set()
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            
            if (x, y) in visited:
                continue
            
            # Check bounds and obstacles
            if (x < 0 or x >= game_state.width or 
                y < 0 or y >= game_state.height or
                (x, y) in game_state.snake):
                continue
            
            visited.add((x, y))
            
            # Add neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((x + dx, y + dy))
        
        return len(visited)
    
    def _calculate_strategic_reward(self, game_state: GameState) -> float:
        """Calculate reward for strategic behavior"""
        strategic_score = 0
        
        if not game_state.snake:
            return 0
        
        head_x, head_y = game_state.snake[0]
        food_x, food_y = game_state.food
        
        # Reward for positioning near center (good strategic position)
        center_x, center_y = game_state.width // 2, game_state.height // 2
        distance_to_center = abs(head_x - center_x) + abs(head_y - center_y)
        max_center_distance = game_state.width // 2 + game_state.height // 2
        
        center_bonus = 1.0 - (distance_to_center / max_center_distance)
        strategic_score += center_bonus * 0.3
        
        # Reward for maintaining good shape (not too coiled)
        shape_score = self._evaluate_snake_shape(game_state)
        strategic_score += shape_score * 0.4
        
        # Reward for food approach strategy
        approach_score = self._evaluate_food_approach(game_state)
        strategic_score += approach_score * 0.3
        
        return strategic_score
    
    def _evaluate_snake_shape(self, game_state: GameState) -> float:
        """Evaluate how well-shaped the snake is"""
        if len(game_state.snake) < 4:
            return 1.0  # Short snake is fine
        
        # Calculate how spread out the snake is
        positions = game_state.snake
        
        # Calculate bounding box
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        
        bounding_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        snake_length = len(positions)
        
        # Good shape has reasonable area per segment
        if bounding_area > 0:
            shape_score = min(1.0, bounding_area / (snake_length * 2))
        else:
            shape_score = 1.0
        
        return shape_score
    
    def _evaluate_food_approach(self, game_state: GameState) -> float:
        """Evaluate the approach strategy toward food"""
        if not game_state.snake:
            return 0
        
        head_x, head_y = game_state.snake[0]
        food_x, food_y = game_state.food
        
        # Check if moving in generally correct direction
        if len(self.path_history) >= 2:
            prev_head = self.path_history[-2]
            prev_distance = abs(prev_head[0] - food_x) + abs(prev_head[1] - food_y)
            current_distance = abs(head_x - food_x) + abs(head_y - food_y)
            
            if current_distance < prev_distance:
                return 1.0  # Moving closer
            elif current_distance > prev_distance:
                return -0.5  # Moving away
        
        # Store current position for next evaluation
        self.path_history.append((head_x, head_y))
        if len(self.path_history) > 10:
            self.path_history.pop(0)
        
        return 0
    
    def _calculate_progress_reward(self, game_state: GameState) -> float:
        """Calculate reward for overall progress"""
        
        # Progress based on score improvement
        score_progress = game_state.score * 0.5
        
        # Progress based on survival time
        survival_progress = min(1.0, game_state.steps / 100) * 0.3
        
        # Progress based on snake length
        length_progress = min(1.0, len(game_state.snake) / 10) * 0.2
        
        return score_progress + survival_progress + length_progress
    
    def adapt_reward_weights(self, recent_performance: List[float]):
        """Adapt reward weights based on recent performance"""
        if len(recent_performance) < 50:
            return
        
        avg_performance = np.mean(recent_performance)
        performance_std = np.std(recent_performance)
        
        # If performance is stagnating, increase exploration
        if performance_std < 0.1:
            self.reward_weights[RewardType.EXPLORATION] *= 1.1
            self.reward_weights[RewardType.STRATEGIC] *= 1.05
        
        # If performance is highly variable, increase safety
        elif performance_std > 0.5:
            self.reward_weights[RewardType.SAFETY] *= 1.1
            self.reward_weights[RewardType.EFFICIENCY] *= 1.05
        
        # Normalize weights to prevent drift
        total_weight = sum(self.reward_weights.values())
        if total_weight > 2.0:
            for reward_type in self.reward_weights:
                self.reward_weights[reward_type] /= (total_weight / 2.0)
    
    def get_reward_analysis(self) -> Dict:
        """Get analysis of reward distribution"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]
        
        analysis = {
            'total_rewards': {
                'mean': np.mean([r['total'] for r in recent_rewards]),
                'std': np.std([r['total'] for r in recent_rewards]),
                'min': np.min([r['total'] for r in recent_rewards]),
                'max': np.max([r['total'] for r in recent_rewards])
            },
            'component_contributions': {},
            'reward_weights': dict(self.reward_weights),
            'exploration_decay': self.exploration_decay
        }
        
        # Analyze component contributions
        for reward_type in RewardType:
            if reward_type in self.reward_weights:
                component_values = []
                for reward_data in recent_rewards:
                    if reward_type in reward_data['components']:
                        weighted_value = (reward_data['components'][reward_type] * 
                                        self.reward_weights[reward_type])
                        component_values.append(weighted_value)
                
                if component_values:
                    analysis['component_contributions'][reward_type.value] = {
                        'mean': np.mean(component_values),
                        'contribution_ratio': np.mean(component_values) / analysis['total_rewards']['mean']
                    }
        
        return analysis
    
    def reset_reward_shaper(self):
        """Reset reward shaper state"""
        self.reward_history = []
        self.state_visit_counts = {}
        self.path_history = []
        
        self.logger.info("Reward shaper reset")
    
    def set_curriculum_adaptation(self, curriculum_stage: str):
        """Adapt reward shaping for specific curriculum stage"""
        
        if curriculum_stage == "Tutorial":
            # Emphasize basic rewards
            self.reward_weights[RewardType.FOOD] = 15.0
            self.reward_weights[RewardType.DISTANCE] = 0.2
            self.reward_weights[RewardType.SAFETY] = 0.05
            
        elif curriculum_stage == "Basic Navigation":
            # Emphasize exploration and safety
            self.reward_weights[RewardType.EXPLORATION] = 0.1
            self.reward_weights[RewardType.SAFETY] = 0.2
            
        elif curriculum_stage == "Self-Avoidance":
            # Emphasize safety heavily
            self.reward_weights[RewardType.SAFETY] = 0.3
            self.reward_weights[RewardType.STRATEGIC] = 0.1
            
        elif curriculum_stage == "Strategic Planning":
            # Emphasize strategic thinking
            self.reward_weights[RewardType.STRATEGIC] = 0.15
            self.reward_weights[RewardType.EFFICIENCY] = 0.05
            
        elif curriculum_stage == "Master Level":
            # Balanced approach with emphasis on efficiency
            self.reward_weights[RewardType.EFFICIENCY] = 0.08
            self.reward_weights[RewardType.STRATEGIC] = 0.12
        
        self.logger.info(f"Adapted reward weights for {curriculum_stage}")

class HierarchicalRewardShaper(RewardShaper):
    """Hierarchical reward shaper with multiple time scales"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Multi-scale reward tracking
        self.short_term_goals = []  # 1-10 steps
        self.medium_term_goals = []  # 10-50 steps
        self.long_term_goals = []   # 50+ steps
        
    def add_hierarchical_rewards(self, game_state: GameState, 
                                action_sequence: List[int]) -> float:
        """Add rewards based on hierarchical goal achievement"""
        
        hierarchical_reward = 0
        
        # Short-term: immediate tactical decisions
        short_term_reward = self._evaluate_short_term_goals(game_state)
        hierarchical_reward += short_term_reward * 0.3
        
        # Medium-term: local strategic objectives
        medium_term_reward = self._evaluate_medium_term_goals(game_state)
        hierarchical_reward += medium_term_reward * 0.5
        
        # Long-term: overall game strategy
        long_term_reward = self._evaluate_long_term_goals(game_state)
        hierarchical_reward += long_term_reward * 0.2
        
        return hierarchical_reward
    
    def _evaluate_short_term_goals(self, game_state: GameState) -> float:
        """Evaluate short-term tactical goals"""
        # Goals: avoid immediate death, move toward food
        return self._calculate_safety_reward(game_state) + self._calculate_distance_reward(game_state)
    
    def _evaluate_medium_term_goals(self, game_state: GameState) -> float:
        """Evaluate medium-term strategic goals"""
        # Goals: maintain good position, efficient food collection
        return self._calculate_strategic_reward(game_state) + self._calculate_efficiency_reward(game_state, {})
    
    def _evaluate_long_term_goals(self, game_state: GameState) -> float:
        """Evaluate long-term strategic goals"""
        # Goals: maximize score, optimize snake shape
        return self._calculate_progress_reward(game_state)

class CuriosityDrivenRewardShaper(RewardShaper):
    """Reward shaper that includes curiosity-driven exploration"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        self.prediction_network = None  # Would be a neural network
        self.prediction_errors = []
        
    def add_curiosity_reward(self, state: np.ndarray, action: int, 
                           next_state: np.ndarray) -> float:
        """Add intrinsic curiosity reward based on prediction error"""
        
        if self.prediction_network is None:
            return 0
        
        # This would use a forward/inverse model to calculate prediction error
        # For now, return a simple novelty-based reward
        
        state_key = tuple(state.flatten())
        novelty = 1.0 / (1.0 + self.state_visit_counts.get(state_key, 0))
        
        return novelty * 0.1
