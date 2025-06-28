"""
Curriculum Learning implementation for progressive Snake AI training
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import random

from game.game_state import GameState
from utils.logger import setup_logger

class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning"""
    BEGINNER = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4

@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage"""
    name: str
    difficulty: DifficultyLevel
    grid_size: int
    episode_length: int
    food_spawn_strategy: str
    obstacle_density: float
    success_threshold: float
    episodes_required: int
    reward_multiplier: float
    description: str

class CurriculumManager:
    """Manages curriculum learning progression"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = setup_logger('curriculum_manager')
        
        # Initialize curriculum stages
        self.stages = self._initialize_stages()
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        
        # Progress tracking
        self.stage_episodes = 0
        self.stage_successes = 0
        self.total_episodes = 0
        self.progression_history = []
        
        # Adaptive parameters
        self.adaptive_thresholds = True
        self.min_episodes_per_stage = 100
        self.max_episodes_per_stage = 2000
        
    def _initialize_stages(self) -> List[CurriculumStage]:
        """Initialize curriculum stages"""
        stages = [
            CurriculumStage(
                name="Tutorial",
                difficulty=DifficultyLevel.BEGINNER,
                grid_size=8,
                episode_length=50,
                food_spawn_strategy="adjacent",
                obstacle_density=0.0,
                success_threshold=0.7,
                episodes_required=200,
                reward_multiplier=1.5,
                description="Learn basic movement and food collection"
            ),
            CurriculumStage(
                name="Basic Navigation",
                difficulty=DifficultyLevel.EASY,
                grid_size=12,
                episode_length=100,
                food_spawn_strategy="nearby",
                obstacle_density=0.0,
                success_threshold=0.6,
                episodes_required=300,
                reward_multiplier=1.2,
                description="Navigate larger spaces and avoid walls"
            ),
            CurriculumStage(
                name="Self-Avoidance",
                difficulty=DifficultyLevel.MEDIUM,
                grid_size=16,
                episode_length=200,
                food_spawn_strategy="random",
                obstacle_density=0.0,
                success_threshold=0.5,
                episodes_required=500,
                reward_multiplier=1.0,
                description="Learn to avoid self-collision with longer snake"
            ),
            CurriculumStage(
                name="Strategic Planning",
                difficulty=DifficultyLevel.HARD,
                grid_size=20,
                episode_length=400,
                food_spawn_strategy="strategic",
                obstacle_density=0.1,
                success_threshold=0.4,
                episodes_required=1000,
                reward_multiplier=0.8,
                description="Plan ahead and navigate around obstacles"
            ),
            CurriculumStage(
                name="Master Level",
                difficulty=DifficultyLevel.EXPERT,
                grid_size=24,
                episode_length=1000,
                food_spawn_strategy="adversarial",
                obstacle_density=0.2,
                success_threshold=0.3,
                episodes_required=2000,
                reward_multiplier=0.6,
                description="Master-level gameplay with complex scenarios"
            )
        ]
        
        return stages
    
    def get_current_stage_config(self) -> Dict:
        """Get configuration for current curriculum stage"""
        return {
            'stage_name': self.current_stage.name,
            'difficulty': self.current_stage.difficulty,
            'grid_size': self.current_stage.grid_size,
            'episode_length': self.current_stage.episode_length,
            'food_spawn_strategy': self.current_stage.food_spawn_strategy,
            'obstacle_density': self.current_stage.obstacle_density,
            'reward_multiplier': self.current_stage.reward_multiplier,
            'stage_progress': self.stage_episodes / self.current_stage.episodes_required,
            'success_rate': self.stage_successes / max(1, self.stage_episodes)
        }
    
    def create_game_environment(self) -> GameState:
        """Create game environment for current curriculum stage"""
        game_state = GameState(
            width=self.current_stage.grid_size,
            height=self.current_stage.grid_size
        )
        
        # Apply stage-specific modifications
        self._apply_stage_modifications(game_state)
        
        return game_state
    
    def _apply_stage_modifications(self, game_state: GameState):
        """Apply curriculum stage modifications to game state"""
        
        # Add obstacles based on difficulty
        if self.current_stage.obstacle_density > 0:
            self._add_obstacles(game_state)
        
        # Modify food placement
        if self.current_stage.food_spawn_strategy != "random":
            self._set_strategic_food_placement(game_state)
    
    def _add_obstacles(self, game_state: GameState):
        """Add obstacles to the game environment"""
        num_obstacles = int(
            game_state.width * game_state.height * self.current_stage.obstacle_density
        )
        
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                x = random.randint(0, game_state.width - 1)
                y = random.randint(0, game_state.height - 1)
                
                # Avoid placing obstacles on snake or food
                if ((x, y) not in game_state.snake and 
                    (x, y) != game_state.food and
                    (x, y) not in obstacles):
                    obstacles.append((x, y))
                    break
        
        # Store obstacles in game state (would need to modify GameState class)
        if hasattr(game_state, 'obstacles'):
            game_state.obstacles = obstacles
    
    def _set_strategic_food_placement(self, game_state: GameState):
        """Set strategic food placement based on curriculum stage"""
        head_x, head_y = game_state.snake[0] if game_state.snake else (0, 0)
        
        if self.current_stage.food_spawn_strategy == "adjacent":
            # Place food adjacent to snake head
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in directions:
                x, y = head_x + dx, head_y + dy
                if (0 <= x < game_state.width and 
                    0 <= y < game_state.height and
                    (x, y) not in game_state.snake):
                    game_state.food = (x, y)
                    break
        
        elif self.current_stage.food_spawn_strategy == "nearby":
            # Place food within 3 tiles of snake head
            for distance in range(1, 4):
                positions = []
                for dx in range(-distance, distance + 1):
                    for dy in range(-distance, distance + 1):
                        if abs(dx) + abs(dy) == distance:
                            x, y = head_x + dx, head_y + dy
                            if (0 <= x < game_state.width and 
                                0 <= y < game_state.height and
                                (x, y) not in game_state.snake):
                                positions.append((x, y))
                
                if positions:
                    game_state.food = random.choice(positions)
                    break
        
        elif self.current_stage.food_spawn_strategy == "strategic":
            # Place food to encourage strategic thinking
            # Prefer corners and edges to force planning
            strategic_positions = []
            
            # Corner positions
            corners = [
                (0, 0), (0, game_state.height - 1),
                (game_state.width - 1, 0), (game_state.width - 1, game_state.height - 1)
            ]
            
            # Edge positions
            for x in range(game_state.width):
                strategic_positions.extend([(x, 0), (x, game_state.height - 1)])
            for y in range(game_state.height):
                strategic_positions.extend([(0, y), (game_state.width - 1, y)])
            
            # Filter valid positions
            valid_positions = [
                pos for pos in strategic_positions
                if pos not in game_state.snake
            ]
            
            if valid_positions:
                game_state.food = random.choice(valid_positions)
        
        elif self.current_stage.food_spawn_strategy == "adversarial":
            # Place food in positions that require complex planning
            # Maximize distance from head while avoiding easy paths
            head_x, head_y = game_state.snake[0]
            
            candidates = []
            for x in range(game_state.width):
                for y in range(game_state.height):
                    if (x, y) not in game_state.snake:
                        distance = abs(x - head_x) + abs(y - head_y)
                        # Prefer distant positions
                        if distance > game_state.width // 2:
                            candidates.append((x, y))
            
            if candidates:
                game_state.food = random.choice(candidates)
    
    def report_episode_result(self, score: int, steps: int, success: bool):
        """Report the result of an episode"""
        self.stage_episodes += 1
        self.total_episodes += 1
        
        if success or score > 0:  # Define success criteria
            self.stage_successes += 1
        
        # Check for stage progression
        if self._should_advance_stage():
            self._advance_to_next_stage()
        
        # Log progress
        if self.stage_episodes % 50 == 0:
            self._log_progress()
    
    def _should_advance_stage(self) -> bool:
        """Check if agent should advance to next stage"""
        if self.stage_episodes < self.min_episodes_per_stage:
            return False
        
        success_rate = self.stage_successes / self.stage_episodes
        
        # Adaptive threshold adjustment
        if self.adaptive_thresholds:
            adjusted_threshold = self._calculate_adaptive_threshold()
        else:
            adjusted_threshold = self.current_stage.success_threshold
        
        # Advance if success threshold met or max episodes reached
        return (success_rate >= adjusted_threshold or 
                self.stage_episodes >= self.max_episodes_per_stage)
    
    def _calculate_adaptive_threshold(self) -> float:
        """Calculate adaptive threshold based on recent performance"""
        base_threshold = self.current_stage.success_threshold
        
        # Look at recent performance trend
        if self.stage_episodes >= 100:
            recent_episodes = min(100, self.stage_episodes)
            recent_successes = sum(1 for _ in range(recent_episodes) 
                                 if random.random() < (self.stage_successes / self.stage_episodes))
            recent_success_rate = recent_successes / recent_episodes
            
            # Adjust threshold based on trend
            if recent_success_rate > base_threshold * 1.2:
                return base_threshold * 0.9  # Make it easier to advance
            elif recent_success_rate < base_threshold * 0.5:
                return base_threshold * 1.1  # Make it harder to advance
        
        return base_threshold
    
    def _advance_to_next_stage(self):
        """Advance to the next curriculum stage"""
        # Record progression
        progression_info = {
            'stage_name': self.current_stage.name,
            'episodes': self.stage_episodes,
            'successes': self.stage_successes,
            'success_rate': self.stage_successes / self.stage_episodes,
            'total_episodes': self.total_episodes
        }
        self.progression_history.append(progression_info)
        
        self.logger.info(
            f"Advancing from {self.current_stage.name} after {self.stage_episodes} episodes "
            f"with {self.stage_successes / self.stage_episodes:.2%} success rate"
        )
        
        # Move to next stage
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.current_stage = self.stages[self.current_stage_idx]
            
            self.logger.info(f"Starting stage: {self.current_stage.name}")
        else:
            self.logger.info("Curriculum completed! Continuing at expert level.")
        
        # Reset stage counters
        self.stage_episodes = 0
        self.stage_successes = 0
    
    def _log_progress(self):
        """Log curriculum progress"""
        success_rate = self.stage_successes / self.stage_episodes
        progress = self.stage_episodes / self.current_stage.episodes_required
        
        self.logger.info(
            f"Stage: {self.current_stage.name} | "
            f"Episodes: {self.stage_episodes}/{self.current_stage.episodes_required} | "
            f"Success Rate: {success_rate:.2%} | "
            f"Progress: {progress:.1%}"
        )
    
    def get_reward_multiplier(self) -> float:
        """Get reward multiplier for current stage"""
        return self.current_stage.reward_multiplier
    
    def get_episode_length_limit(self) -> int:
        """Get episode length limit for current stage"""
        return self.current_stage.episode_length
    
    def is_curriculum_complete(self) -> bool:
        """Check if curriculum is complete"""
        return self.current_stage_idx >= len(self.stages) - 1
    
    def get_curriculum_statistics(self) -> Dict:
        """Get comprehensive curriculum statistics"""
        current_success_rate = (self.stage_successes / max(1, self.stage_episodes))
        
        stats = {
            'current_stage': {
                'name': self.current_stage.name,
                'difficulty': self.current_stage.difficulty.name,
                'episodes': self.stage_episodes,
                'successes': self.stage_successes,
                'success_rate': current_success_rate,
                'progress': self.stage_episodes / self.current_stage.episodes_required,
                'target_threshold': self.current_stage.success_threshold
            },
            'overall_progress': {
                'total_episodes': self.total_episodes,
                'stages_completed': len(self.progression_history),
                'current_stage_index': self.current_stage_idx,
                'total_stages': len(self.stages),
                'curriculum_completion': self.current_stage_idx / (len(self.stages) - 1)
            },
            'progression_history': self.progression_history
        }
        
        return stats
    
    def reset_curriculum(self):
        """Reset curriculum to beginning"""
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        self.stage_episodes = 0
        self.stage_successes = 0
        self.total_episodes = 0
        self.progression_history = []
        
        self.logger.info("Curriculum reset to beginning")
    
    def skip_to_stage(self, stage_name: str) -> bool:
        """Skip to a specific curriculum stage"""
        for idx, stage in enumerate(self.stages):
            if stage.name == stage_name:
                self.current_stage_idx = idx
                self.current_stage = stage
                self.stage_episodes = 0
                self.stage_successes = 0
                
                self.logger.info(f"Skipped to stage: {stage_name}")
                return True
        
        self.logger.warning(f"Stage not found: {stage_name}")
        return False

class AdaptiveCurriculum:
    """Adaptive curriculum that adjusts based on agent performance"""
    
    def __init__(self, base_curriculum: CurriculumManager):
        self.base_curriculum = base_curriculum
        self.performance_history = []
        self.adaptation_frequency = 100  # Episodes between adaptations
        
    def adapt_curriculum(self, recent_performance: List[float]):
        """Adapt curriculum based on recent performance"""
        if len(recent_performance) < 10:
            return
        
        avg_performance = np.mean(recent_performance)
        performance_std = np.std(recent_performance)
        
        current_stage = self.base_curriculum.current_stage
        
        # Adjust difficulty based on performance
        if avg_performance > current_stage.success_threshold * 1.5:
            # Agent is doing too well, increase difficulty
            self._increase_difficulty()
        elif avg_performance < current_stage.success_threshold * 0.5:
            # Agent is struggling, decrease difficulty
            self._decrease_difficulty()
        
        # Adjust stage duration based on learning stability
        if performance_std < 0.1:  # Stable learning
            # Can advance faster
            current_stage.episodes_required = max(
                current_stage.episodes_required * 0.9,
                100
            )
        elif performance_std > 0.3:  # Unstable learning
            # Need more time
            current_stage.episodes_required = min(
                current_stage.episodes_required * 1.1,
                2000
            )
    
    def _increase_difficulty(self):
        """Increase difficulty of current stage"""
        stage = self.base_curriculum.current_stage
        
        # Increase grid size
        if stage.grid_size < 30:
            stage.grid_size += 2
        
        # Increase obstacle density
        if stage.obstacle_density < 0.3:
            stage.obstacle_density += 0.05
        
        # Decrease reward multiplier
        if stage.reward_multiplier > 0.5:
            stage.reward_multiplier *= 0.95
    
    def _decrease_difficulty(self):
        """Decrease difficulty of current stage"""
        stage = self.base_curriculum.current_stage
        
        # Decrease grid size
        if stage.grid_size > 8:
            stage.grid_size -= 1
        
        # Decrease obstacle density
        if stage.obstacle_density > 0:
            stage.obstacle_density = max(0, stage.obstacle_density - 0.02)
        
        # Increase reward multiplier
        if stage.reward_multiplier < 2.0:
            stage.reward_multiplier *= 1.05

class CurriculumEvaluator:
    """Evaluates agent performance across curriculum stages"""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_stage_performance(self, stage_name: str, 
                                 performance_metrics: Dict) -> Dict:
        """Evaluate performance on a specific curriculum stage"""
        
        evaluation = {
            'stage_name': stage_name,
            'timestamp': np.datetime64('now'),
            'metrics': performance_metrics,
            'score': self._calculate_stage_score(performance_metrics),
            'recommendations': self._generate_recommendations(performance_metrics)
        }
        
        self.evaluation_results[stage_name] = evaluation
        return evaluation
    
    def _calculate_stage_score(self, metrics: Dict) -> float:
        """Calculate overall score for stage performance"""
        weights = {
            'success_rate': 0.4,
            'average_score': 0.3,
            'efficiency': 0.2,
            'consistency': 0.1
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                normalized_value = min(1.0, metrics[metric] / 100)  # Normalize to [0,1]
                score += weight * normalized_value
        
        return score
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on performance"""
        recommendations = []
        
        if metrics.get('success_rate', 0) < 0.3:
            recommendations.append("Consider reducing difficulty or extending training time")
        
        if metrics.get('efficiency', 0) < 0.5:
            recommendations.append("Focus on path planning and strategic thinking")
        
        if metrics.get('consistency', 0) < 0.6:
            recommendations.append("Increase training stability with regularization")
        
        return recommendations
