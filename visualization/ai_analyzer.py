"""
AI behavior analysis and visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import time

from game.game_state import GameState
from utils.logger import setup_logger

class AIAnalyzer:
    """Advanced AI behavior analyzer"""
    
    def __init__(self):
        self.logger = setup_logger('ai_analyzer')
        
        # Analysis data storage
        self.decision_history = []
        self.state_embeddings = []
        self.performance_metrics = []
        self.attention_patterns = []
        
        # Behavioral patterns
        self.strategy_patterns = {}
        self.decision_trees = {}
        self.learning_curves = {}
        
        # Visualization settings
        plt.style.use('dark_background')
        sns.set_palette("husl")
    
    def analyze_decision_making(self, game_state: GameState, 
                              ai_outputs: Dict, action_taken: int) -> Dict:
        """Analyze AI decision-making process"""
        
        analysis = {
            'timestamp': time.time(),
            'game_state': self._extract_state_features(game_state),
            'ai_outputs': ai_outputs,
            'action_taken': action_taken,
            'decision_quality': self._evaluate_decision_quality(game_state, action_taken),
            'strategic_assessment': self._assess_strategy(game_state, ai_outputs),
            'attention_analysis': self._analyze_attention_patterns(ai_outputs),
            'confidence_metrics': self._calculate_confidence_metrics(ai_outputs)
        }
        
        # Store for pattern analysis
        self.decision_history.append(analysis)
        
        # Keep only recent decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        
        return analysis
    
    def _extract_state_features(self, game_state: GameState) -> Dict:
        """Extract relevant features from game state"""
        
        if not game_state.snake:
            return {}
        
        head_x, head_y = game_state.snake[0]
        food_x, food_y = game_state.food
        
        features = {
            'snake_length': len(game_state.snake),
            'head_position': (head_x, head_y),
            'food_position': (food_x, food_y),
            'food_distance': abs(head_x - food_x) + abs(head_y - food_y),
            'wall_distances': self._calculate_wall_distances(game_state),
            'body_proximity': self._calculate_body_proximity(game_state),
            'available_space': self._calculate_available_space(game_state),
            'board_coverage': len(game_state.snake) / (game_state.width * game_state.height),
            'game_progress': game_state.steps,
            'score': game_state.score
        }
        
        return features
    
    def _calculate_wall_distances(self, game_state: GameState) -> Dict:
        """Calculate distances to walls in all directions"""
        
        if not game_state.snake:
            return {}
        
        head_x, head_y = game_state.snake[0]
        
        return {
            'up': head_y,
            'down': game_state.height - head_y - 1,
            'left': head_x,
            'right': game_state.width - head_x - 1
        }
    
    def _calculate_body_proximity(self, game_state: GameState) -> Dict:
        """Calculate proximity to snake body segments"""
        
        if len(game_state.snake) < 2:
            return {'min_distance': float('inf'), 'avg_distance': float('inf')}
        
        head_x, head_y = game_state.snake[0]
        body_segments = game_state.snake[1:]
        
        distances = []
        for seg_x, seg_y in body_segments:
            distance = abs(head_x - seg_x) + abs(head_y - seg_y)
            distances.append(distance)
        
        return {
            'min_distance': min(distances),
            'avg_distance': np.mean(distances),
            'segments_nearby': sum(1 for d in distances if d <= 2)
        }
    
    def _calculate_available_space(self, game_state: GameState) -> int:
        """Calculate available space using flood fill"""
        
        if not game_state.snake:
            return game_state.width * game_state.height
        
        head_x, head_y = game_state.snake[0]
        visited = set()
        stack = [(head_x, head_y)]
        
        while stack:
            x, y = stack.pop()
            
            if (x, y) in visited:
                continue
            
            if (x < 0 or x >= game_state.width or 
                y < 0 or y >= game_state.height or
                (x, y) in game_state.snake):
                continue
            
            visited.add((x, y))
            
            # Add neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((x + dx, y + dy))
        
        return len(visited)
    
    def _evaluate_decision_quality(self, game_state: GameState, action: int) -> Dict:
        """Evaluate the quality of the decision made"""
        
        # Simulate the action to see immediate consequences
        temp_state = GameState(game_state.width, game_state.height)
        temp_state.snake = game_state.snake.copy()
        temp_state.direction = game_state.direction
        temp_state.food = game_state.food
        temp_state.score = game_state.score
        temp_state.steps = game_state.steps
        
        reward, done = temp_state.step(action)
        
        quality_metrics = {
            'immediate_reward': reward,
            'survives_action': not done,
            'moves_toward_food': self._check_food_approach(game_state, action),
            'avoids_walls': self._check_wall_avoidance(game_state, action),
            'avoids_body': self._check_body_avoidance(game_state, action),
            'maintains_space': self._check_space_maintenance(game_state, action),
            'overall_score': 0.0
        }
        
        # Calculate overall quality score
        weights = {
            'survives_action': 5.0,
            'moves_toward_food': 2.0,
            'avoids_walls': 3.0,
            'avoids_body': 4.0,
            'maintains_space': 2.0
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, value in quality_metrics.items():
            if metric in weights and isinstance(value, bool):
                total_score += weights[metric] * (1 if value else 0)
                total_weight += weights[metric]
        
        if total_weight > 0:
            quality_metrics['overall_score'] = total_score / total_weight
        
        return quality_metrics
    
    def _check_food_approach(self, game_state: GameState, action: int) -> bool:
        """Check if action moves toward food"""
        
        if not game_state.snake:
            return False
        
        head_x, head_y = game_state.snake[0]
        food_x, food_y = game_state.food
        
        # Calculate new head position
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
        if action < len(directions):
            dx, dy = directions[action]
            new_x, new_y = head_x + dx, head_y + dy
            
            # Check if distance to food decreases
            old_distance = abs(head_x - food_x) + abs(head_y - food_y)
            new_distance = abs(new_x - food_x) + abs(new_y - food_y)
            
            return new_distance < old_distance
        
        return False
    
    def _check_wall_avoidance(self, game_state: GameState, action: int) -> bool:
        """Check if action avoids walls"""
        
        if not game_state.snake:
            return True
        
        head_x, head_y = game_state.snake[0]
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        if action < len(directions):
            dx, dy = directions[action]
            new_x, new_y = head_x + dx, head_y + dy
            
            return (0 <= new_x < game_state.width and 
                   0 <= new_y < game_state.height)
        
        return False
    
    def _check_body_avoidance(self, game_state: GameState, action: int) -> bool:
        """Check if action avoids snake body"""
        
        if not game_state.snake:
            return True
        
        head_x, head_y = game_state.snake[0]
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        if action < len(directions):
            dx, dy = directions[action]
            new_x, new_y = head_x + dx, head_y + dy
            
            return (new_x, new_y) not in game_state.snake
        
        return False
    
    def _check_space_maintenance(self, game_state: GameState, action: int) -> bool:
        """Check if action maintains adequate space"""
        
        # Simple heuristic: avoid moves that significantly reduce available space
        current_space = self._calculate_available_space(game_state)
        
        # Simulate action
        temp_state = GameState(game_state.width, game_state.height)
        temp_state.snake = game_state.snake.copy()
        temp_state.direction = game_state.direction
        temp_state.food = game_state.food
        
        reward, done = temp_state.step(action)
        
        if done:
            return False
        
        new_space = self._calculate_available_space(temp_state)
        space_ratio = new_space / max(1, current_space)
        
        return space_ratio > 0.8  # Don't lose more than 20% of space
    
    def _assess_strategy(self, game_state: GameState, ai_outputs: Dict) -> Dict:
        """Assess the strategic approach of the AI"""
        
        strategy_assessment = {
            'strategy_type': 'unknown',
            'risk_level': 'medium',
            'planning_horizon': 'short',
            'exploration_vs_exploitation': 0.5,
            'adaptability_score': 0.5
        }
        
        # Analyze strategy type based on decision patterns
        if 'action_probabilities' in ai_outputs:
            probs = ai_outputs['action_probabilities']
            
            # Check if AI is being aggressive (high-risk, high-reward)
            max_prob = max(probs)
            entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
            
            if max_prob > 0.8:
                strategy_assessment['strategy_type'] = 'confident'
            elif entropy > 1.2:
                strategy_assessment['strategy_type'] = 'exploratory'
            else:
                strategy_assessment['strategy_type'] = 'balanced'
            
            # Risk assessment
            if self._is_risky_position(game_state):
                if max_prob > 0.9:
                    strategy_assessment['risk_level'] = 'high'
                else:
                    strategy_assessment['risk_level'] = 'medium'
            else:
                strategy_assessment['risk_level'] = 'low'
        
        return strategy_assessment
    
    def _is_risky_position(self, game_state: GameState) -> bool:
        """Determine if current position is risky"""
        
        if not game_state.snake:
            return False
        
        head_x, head_y = game_state.snake[0]
        risk_factors = 0
        
        # Check proximity to walls
        if head_x <= 1 or head_x >= game_state.width - 2:
            risk_factors += 1
        if head_y <= 1 or head_y >= game_state.height - 2:
            risk_factors += 1
        
        # Check proximity to body
        body_proximity = self._calculate_body_proximity(game_state)
        if body_proximity.get('min_distance', float('inf')) <= 2:
            risk_factors += 1
        
        # Check available space
        available_space = self._calculate_available_space(game_state)
        total_space = game_state.width * game_state.height
        if available_space / total_space < 0.3:
            risk_factors += 1
        
        return risk_factors >= 2
    
    def _analyze_attention_patterns(self, ai_outputs: Dict) -> Dict:
        """Analyze attention patterns from AI outputs"""
        
        attention_analysis = {
            'attention_focus': 'unknown',
            'attention_distribution': 'uniform',
            'spatial_attention_peaks': [],
            'temporal_attention_consistency': 0.5
        }
        
        if 'attention_weights' in ai_outputs:
            weights = ai_outputs['attention_weights']
            
            # Find attention peaks
            if isinstance(weights, np.ndarray) and weights.ndim == 2:
                # Spatial attention
                peak_indices = np.unravel_index(np.argmax(weights), weights.shape)
                attention_analysis['spatial_attention_peaks'] = [peak_indices]
                
                # Analyze distribution
                max_attention = np.max(weights)
                mean_attention = np.mean(weights)
                
                if max_attention > mean_attention * 3:
                    attention_analysis['attention_distribution'] = 'focused'
                elif max_attention < mean_attention * 1.5:
                    attention_analysis['attention_distribution'] = 'diffuse'
                else:
                    attention_analysis['attention_distribution'] = 'balanced'
        
        # Store for temporal analysis
        self.attention_patterns.append(attention_analysis)
        
        return attention_analysis
    
    def _calculate_confidence_metrics(self, ai_outputs: Dict) -> Dict:
        """Calculate confidence metrics from AI outputs"""
        
        confidence_metrics = {
            'decision_confidence': 0.5,
            'prediction_uncertainty': 0.5,
            'model_agreement': 0.5,
            'consistency_score': 0.5
        }
        
        if 'action_probabilities' in ai_outputs:
            probs = ai_outputs['action_probabilities']
            
            # Decision confidence (max probability)
            confidence_metrics['decision_confidence'] = max(probs)
            
            # Prediction uncertainty (entropy)
            entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
            max_entropy = np.log(len(probs))
            confidence_metrics['prediction_uncertainty'] = entropy / max_entropy
        
        if 'ensemble_predictions' in ai_outputs:
            predictions = ai_outputs['ensemble_predictions']
            
            # Model agreement (variance of predictions)
            if len(predictions) > 1:
                agreement_scores = []
                for action in range(len(predictions[0])):
                    action_probs = [pred[action] for pred in predictions]
                    variance = np.var(action_probs)
                    agreement_scores.append(1.0 - variance)
                
                confidence_metrics['model_agreement'] = np.mean(agreement_scores)
        
        return confidence_metrics
    
    def generate_decision_explanation(self, game_state: GameState, 
                                    ai_outputs: Dict, action_taken: int) -> str:
        """Generate human-readable explanation of AI decision"""
        
        explanations = []
        
        # Basic action description
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        if action_taken < len(action_names):
            explanations.append(f"AI chose to move {action_names[action_taken]}")
        
        # Confidence level
        if 'action_probabilities' in ai_outputs:
            probs = ai_outputs['action_probabilities']
            if action_taken < len(probs):
                confidence = probs[action_taken]
                if confidence > 0.8:
                    explanations.append("with high confidence")
                elif confidence > 0.5:
                    explanations.append("with moderate confidence")
                else:
                    explanations.append("with low confidence")
        
        # Decision reasoning
        decision_quality = self._evaluate_decision_quality(game_state, action_taken)
        
        if decision_quality['moves_toward_food']:
            explanations.append("to approach the food")
        
        if not decision_quality['survives_action']:
            explanations.append("WARNING: This move leads to death!")
        elif decision_quality['avoids_walls'] and decision_quality['avoids_body']:
            explanations.append("while staying safe")
        
        if decision_quality['maintains_space']:
            explanations.append("and maintaining escape routes")
        
        # Strategy assessment
        strategy = self._assess_strategy(game_state, ai_outputs)
        if strategy['risk_level'] == 'high':
            explanations.append("This is a high-risk move")
        elif strategy['strategy_type'] == 'exploratory':
            explanations.append("The AI is exploring options")
        
        return ". ".join(explanations) + "."
    
    def analyze_learning_progress(self, training_data: List[Dict]) -> Dict:
        """Analyze learning progress over time"""
        
        if not training_data:
            return {}
        
        analysis = {
            'performance_trend': self._calculate_performance_trend(training_data),
            'decision_quality_improvement': self._analyze_decision_improvement(training_data),
            'strategy_evolution': self._analyze_strategy_evolution(training_data),
            'consistency_metrics': self._calculate_consistency_metrics(training_data),
            'learning_efficiency': self._calculate_learning_efficiency(training_data)
        }
        
        return analysis
    
    def _calculate_performance_trend(self, training_data: List[Dict]) -> Dict:
        """Calculate performance trend metrics"""
        
        scores = [episode.get('score', 0) for episode in training_data]
        episodes = list(range(len(scores)))
        
        if len(scores) < 2:
            return {'trend': 'insufficient_data'}
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(episodes, scores)
        
        # Recent vs early performance
        early_performance = np.mean(scores[:len(scores)//4]) if len(scores) >= 4 else scores[0]
        recent_performance = np.mean(scores[-len(scores)//4:]) if len(scores) >= 4 else scores[-1]
        
        return {
            'trend': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
            'slope': slope,
            'r_squared': r_value**2,
            'early_performance': early_performance,
            'recent_performance': recent_performance,
            'improvement_ratio': recent_performance / max(1, early_performance)
        }
    
    def _analyze_decision_improvement(self, training_data: List[Dict]) -> Dict:
        """Analyze improvement in decision quality"""
        
        decision_qualities = []
        for episode in training_data:
            if 'decision_analysis' in episode:
                quality = episode['decision_analysis'].get('overall_score', 0)
                decision_qualities.append(quality)
        
        if len(decision_qualities) < 10:
            return {'improvement': 'insufficient_data'}
        
        # Compare early vs recent decision quality
        early_quality = np.mean(decision_qualities[:len(decision_qualities)//4])
        recent_quality = np.mean(decision_qualities[-len(decision_qualities)//4:])
        
        return {
            'early_decision_quality': early_quality,
            'recent_decision_quality': recent_quality,
            'improvement': recent_quality - early_quality,
            'quality_trend': 'improving' if recent_quality > early_quality else 'declining'
        }
    
    def _analyze_strategy_evolution(self, training_data: List[Dict]) -> Dict:
        """Analyze how strategy has evolved"""
        
        strategy_types = []
        risk_levels = []
        
        for episode in training_data:
            if 'strategy_assessment' in episode:
                strategy = episode['strategy_assessment']
                strategy_types.append(strategy.get('strategy_type', 'unknown'))
                risk_levels.append(strategy.get('risk_level', 'medium'))
        
        if not strategy_types:
            return {'evolution': 'no_data'}
        
        # Analyze strategy distribution over time
        recent_strategies = strategy_types[-len(strategy_types)//4:] if len(strategy_types) >= 4 else strategy_types
        strategy_distribution = {
            strategy: recent_strategies.count(strategy) / len(recent_strategies)
            for strategy in set(recent_strategies)
        }
        
        return {
            'dominant_strategy': max(strategy_distribution.items(), key=lambda x: x[1])[0],
            'strategy_distribution': strategy_distribution,
            'risk_evolution': self._analyze_risk_evolution(risk_levels)
        }
    
    def _analyze_risk_evolution(self, risk_levels: List[str]) -> Dict:
        """Analyze how risk-taking behavior has evolved"""
        
        if not risk_levels:
            return {'evolution': 'no_data'}
        
        risk_scores = {'low': 1, 'medium': 2, 'high': 3}
        risk_values = [risk_scores.get(level, 2) for level in risk_levels]
        
        early_risk = np.mean(risk_values[:len(risk_values)//4]) if len(risk_values) >= 4 else risk_values[0]
        recent_risk = np.mean(risk_values[-len(risk_values)//4:]) if len(risk_values) >= 4 else risk_values[-1]
        
        return {
            'early_risk_level': early_risk,
            'recent_risk_level': recent_risk,
            'risk_change': recent_risk - early_risk,
            'risk_evolution': 'more_cautious' if recent_risk < early_risk else 'more_aggressive' if recent_risk > early_risk else 'stable'
        }
    
    def _calculate_consistency_metrics(self, training_data: List[Dict]) -> Dict:
        """Calculate consistency metrics"""
        
        scores = [episode.get('score', 0) for episode in training_data]
        
        if len(scores) < 10:
            return {'consistency': 'insufficient_data'}
        
        # Performance variance
        performance_variance = np.var(scores)
        performance_std = np.std(scores)
        mean_score = np.mean(scores)
        
        # Coefficient of variation
        cv = performance_std / max(1, mean_score)
        
        return {
            'performance_variance': performance_variance,
            'coefficient_of_variation': cv,
            'consistency_rating': 'high' if cv < 0.3 else 'medium' if cv < 0.6 else 'low'
        }
    
    def _calculate_learning_efficiency(self, training_data: List[Dict]) -> Dict:
        """Calculate learning efficiency metrics"""
        
        scores = [episode.get('score', 0) for episode in training_data]
        
        if len(scores) < 50:
            return {'efficiency': 'insufficient_data'}
        
        # Time to reach milestones
        milestones = [1, 5, 10, 20]
        milestone_episodes = {}
        
        for milestone in milestones:
            for i, score in enumerate(scores):
                if score >= milestone:
                    milestone_episodes[milestone] = i
                    break
        
        # Learning rate (episodes needed for improvement)
        improvement_rates = []
        window_size = 10
        
        for i in range(window_size, len(scores)):
            current_window = scores[i-window_size:i]
            previous_window = scores[max(0, i-2*window_size):i-window_size]
            
            if previous_window:
                current_avg = np.mean(current_window)
                previous_avg = np.mean(previous_window)
                improvement_rates.append(current_avg - previous_avg)
        
        return {
            'milestone_episodes': milestone_episodes,
            'average_improvement_rate': np.mean(improvement_rates) if improvement_rates else 0,
            'learning_efficiency_rating': self._rate_learning_efficiency(milestone_episodes, improvement_rates)
        }
    
    def _rate_learning_efficiency(self, milestone_episodes: Dict, improvement_rates: List) -> str:
        """Rate learning efficiency"""
        
        if not milestone_episodes or not improvement_rates:
            return 'unknown'
        
        # Quick milestones and positive improvement rate indicate efficient learning
        avg_milestone_episode = np.mean(list(milestone_episodes.values()))
        avg_improvement = np.mean(improvement_rates)
        
        if avg_milestone_episode < 100 and avg_improvement > 0.1:
            return 'high'
        elif avg_milestone_episode < 200 and avg_improvement > 0:
            return 'medium'
        else:
            return 'low'
    
    def create_behavior_profile(self) -> Dict:
        """Create comprehensive behavior profile"""
        
        if not self.decision_history:
            return {'profile': 'no_data'}
        
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        
        # Extract behavioral patterns
        strategies = [d['strategic_assessment']['strategy_type'] for d in recent_decisions 
                     if 'strategic_assessment' in d]
        risk_levels = [d['strategic_assessment']['risk_level'] for d in recent_decisions 
                      if 'strategic_assessment' in d]
        decision_qualities = [d['decision_quality']['overall_score'] for d in recent_decisions 
                            if 'decision_quality' in d]
        
        profile = {
            'dominant_strategy': max(set(strategies), key=strategies.count) if strategies else 'unknown',
            'preferred_risk_level': max(set(risk_levels), key=risk_levels.count) if risk_levels else 'unknown',
            'average_decision_quality': np.mean(decision_qualities) if decision_qualities else 0,
            'behavioral_consistency': self._calculate_behavioral_consistency(recent_decisions),
            'adaptation_speed': self._calculate_adaptation_speed(recent_decisions),
            'exploration_tendency': self._calculate_exploration_tendency(recent_decisions)
        }
        
        return profile
    
    def _calculate_behavioral_consistency(self, decisions: List[Dict]) -> float:
        """Calculate how consistent the AI's behavior is"""
        
        if len(decisions) < 10:
            return 0.5
        
        strategies = [d['strategic_assessment']['strategy_type'] for d in decisions 
                     if 'strategic_assessment' in d]
        
        if not strategies:
            return 0.5
        
        # Measure how often the strategy changes
        strategy_changes = sum(1 for i in range(1, len(strategies)) 
                             if strategies[i] != strategies[i-1])
        
        consistency = 1.0 - (strategy_changes / max(1, len(strategies) - 1))
        return consistency
    
    def _calculate_adaptation_speed(self, decisions: List[Dict]) -> float:
        """Calculate how quickly AI adapts to new situations"""
        
        # This is a simplified metric - in practice, you'd analyze response to
        # specific situation changes
        decision_qualities = [d['decision_quality']['overall_score'] for d in decisions 
                            if 'decision_quality' in d]
        
        if len(decision_qualities) < 10:
            return 0.5
        
        # Look for improvement patterns after poor decisions
        adaptation_scores = []
        
        for i in range(len(decision_qualities) - 5):
            if decision_qualities[i] < 0.3:  # Poor decision
                # Check if quality improves in next few decisions
                following_qualities = decision_qualities[i+1:i+6]
                improvement = np.mean(following_qualities) - decision_qualities[i]
                adaptation_scores.append(max(0, improvement))
        
        return np.mean(adaptation_scores) if adaptation_scores else 0.5
    
    def _calculate_exploration_tendency(self, decisions: List[Dict]) -> float:
        """Calculate tendency to explore vs exploit"""
        
        strategies = [d['strategic_assessment']['strategy_type'] for d in decisions 
                     if 'strategic_assessment' in d]
        
        if not strategies:
            return 0.5
        
        exploratory_count = strategies.count('exploratory')
        total_strategies = len(strategies)
        
        return exploratory_count / total_strategies
    
    def export_analysis_report(self, filename: str) -> bool:
        """Export comprehensive analysis report"""
        
        try:
            report = {
                'timestamp': time.time(),
                'total_decisions_analyzed': len(self.decision_history),
                'behavior_profile': self.create_behavior_profile(),
                'recent_performance': self._analyze_recent_performance(),
                'learning_insights': self._generate_learning_insights(),
                'recommendations': self._generate_recommendations()
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Analysis report exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export analysis report: {e}")
            return False
    
    def _analyze_recent_performance(self) -> Dict:
        """Analyze recent performance metrics"""
        
        if not self.decision_history:
            return {}
        
        recent_decisions = self.decision_history[-50:]
        
        qualities = [d['decision_quality']['overall_score'] for d in recent_decisions 
                    if 'decision_quality' in d]
        
        return {
            'average_quality': np.mean(qualities) if qualities else 0,
            'quality_trend': self._calculate_trend(qualities),
            'consistency': np.std(qualities) if qualities else 0,
            'peak_quality': max(qualities) if qualities else 0
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        
        if len(values) < 5:
            return 'insufficient_data'
        
        early_avg = np.mean(values[:len(values)//2])
        recent_avg = np.mean(values[len(values)//2:])
        
        if recent_avg > early_avg * 1.1:
            return 'improving'
        elif recent_avg < early_avg * 0.9:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_learning_insights(self) -> List[str]:
        """Generate insights about learning patterns"""
        
        insights = []
        
        if not self.decision_history:
            insights.append("Insufficient data for learning analysis")
            return insights
        
        profile = self.create_behavior_profile()
        
        # Strategy insights
        if profile['dominant_strategy'] == 'exploratory':
            insights.append("AI shows high exploration tendency, good for discovering new strategies")
        elif profile['dominant_strategy'] == 'confident':
            insights.append("AI demonstrates confident decision-making, indicating learned patterns")
        
        # Risk insights
        if profile['preferred_risk_level'] == 'high':
            insights.append("AI tends to take high risks - may need safety training")
        elif profile['preferred_risk_level'] == 'low':
            insights.append("AI is risk-averse - might miss opportunities for better scores")
        
        # Quality insights
        if profile['average_decision_quality'] > 0.7:
            insights.append("High average decision quality indicates effective learning")
        elif profile['average_decision_quality'] < 0.3:
            insights.append("Low decision quality suggests need for additional training")
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving AI performance"""
        
        recommendations = []
        
        if not self.decision_history:
            recommendations.append("Collect more training data for analysis")
            return recommendations
        
        profile = self.create_behavior_profile()
        recent_performance = self._analyze_recent_performance()
        
        # Performance-based recommendations
        if recent_performance.get('quality_trend') == 'declining':
            recommendations.append("Consider adjusting learning rate or reward structure")
        
        if profile['behavioral_consistency'] < 0.3:
            recommendations.append("Improve training stability to reduce erratic behavior")
        
        if profile['adaptation_speed'] < 0.3:
            recommendations.append("Enhance curriculum learning to improve adaptation")
        
        # Strategy-based recommendations
        if profile['exploration_tendency'] > 0.8:
            recommendations.append("Reduce exploration rate to focus on exploitation")
        elif profile['exploration_tendency'] < 0.2:
            recommendations.append("Increase exploration to discover new strategies")
        
        if profile['average_decision_quality'] < 0.5:
            recommendations.append("Focus on safety-oriented reward shaping")
        
        return recommendations

