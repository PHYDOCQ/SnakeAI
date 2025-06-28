"""
Advanced trainer for Snake AI with multiple architectures and training strategies
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import os
from collections import deque
import threading
import queue
import json

from game.snake_game import SnakeGame
from ai.agents.rainbow_agent import RainbowAgent
from ai.agents.mcts_agent import MCTSAgent
from ai.models.ensemble_model import HybridEnsemble
from ai.training.curriculum_learning import CurriculumManager
from ai.utils.reward_engineering import RewardShaper
from ai.utils.preprocessing import StatePreprocessor
from visualization.ai_analyzer import AIAnalyzer
from utils.logger import TrainingLogger, setup_logger

class AdvancedTrainer:
    """Advanced trainer for Snake AI with multiple training strategies"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = setup_logger('trainer')
        self.training_logger = TrainingLogger('training')
        
        # Training control
        self.is_training = False
        self.training_thread = None
        self.stop_training = threading.Event()
        
        # Initialize components
        self._initialize_components()
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.best_score = 0
        self.recent_scores = deque(maxlen=100)
        self.training_start_time = None
        
        # Performance tracking
        self.performance_metrics = {
            'scores': [],
            'episode_lengths': [],
            'losses': [],
            'q_values': [],
            'exploration_rates': [],
            'curriculum_progress': []
        }
        
        # Model comparison
        self.agent_performance = {
            'rainbow': {'scores': [], 'wins': 0},
            'mcts': {'scores': [], 'wins': 0},
            'ensemble': {'scores': [], 'wins': 0}
        }
    
    def _initialize_components(self):
        """Initialize all training components"""
        
        # Game environment
        self.game = SnakeGame(self.config, headless=True)
        
        # State preprocessor
        self.preprocessor = StatePreprocessor(self.config)
        
        # Reward shaper
        self.reward_shaper = RewardShaper(self.config)
        
        # Curriculum learning
        self.curriculum = CurriculumManager(self.config)
        
        # AI agents
        self._initialize_agents()
        
        # AI analyzer
        self.ai_analyzer = AIAnalyzer()
        
        # Training strategies
        self.training_strategies = ['rainbow', 'mcts', 'ensemble', 'competitive']
        self.current_strategy = 'rainbow'
        
    def _initialize_agents(self):
        """Initialize AI agents"""
        
        # Get state dimensions
        sample_state = self.game.get_compact_state()
        compact_state_size = len(sample_state)
        
        sample_cnn_state = self.game.get_state()
        cnn_input_shape = sample_cnn_state.shape
        
        # Rainbow DQN agent
        self.rainbow_agent = RainbowAgent(
            state_size=compact_state_size,
            action_size=4,
            config=self.config
        )
        
        # MCTS agent
        self.mcts_agent = MCTSAgent(
            simulations=self.config.MCTS_SIMULATIONS,
            c_puct=self.config.MCTS_C_PUCT,
            config=self.config
        )
        
        # Hybrid ensemble
        self.ensemble_agent = HybridEnsemble(
            compact_input_size=compact_state_size,
            cnn_input_shape=cnn_input_shape
        ).to(self.device)
        
        # Set MCTS networks (optional neural network guidance)
        self.mcts_agent.set_networks(
            value_network=self.rainbow_agent.q_network,
            policy_network=self.rainbow_agent.q_network
        )
    
    def start_training(self):
        """Start training in a separate thread"""
        if not self.is_training:
            self.is_training = True
            self.stop_training.clear()
            self.training_start_time = time.time()
            
            self.training_thread = threading.Thread(
                target=self._training_loop,
                daemon=True
            )
            self.training_thread.start()
            
            self.logger.info("Training started")
    
    def stop_training_process(self):
        """Stop the training process"""
        if self.is_training:
            self.stop_training.set()
            self.is_training = False
            
            if self.training_thread:
                self.training_thread.join(timeout=5.0)
            
            self.logger.info("Training stopped")
    
    def _training_loop(self):
        """Main training loop"""
        
        try:
            while not self.stop_training.is_set():
                
                # Get curriculum configuration
                curriculum_config = self.curriculum.get_current_stage_config()
                
                # Run training episode based on current strategy
                if self.current_strategy == 'rainbow':
                    episode_result = self._train_rainbow_episode(curriculum_config)
                elif self.current_strategy == 'mcts':
                    episode_result = self._train_mcts_episode(curriculum_config)
                elif self.current_strategy == 'ensemble':
                    episode_result = self._train_ensemble_episode(curriculum_config)
                elif self.current_strategy == 'competitive':
                    episode_result = self._train_competitive_episode(curriculum_config)
                
                # Update curriculum
                self._update_curriculum(episode_result)
                
                # Update performance tracking
                self._update_performance_metrics(episode_result)
                
                # Save models periodically
                if self.episode_count % self.config.SAVE_INTERVAL == 0:
                    self._save_models()
                
                # Evaluate models periodically
                if self.episode_count % self.config.EVALUATION_INTERVAL == 0:
                    self._evaluate_models()
                
                # Switch training strategies periodically
                if self.episode_count % 500 == 0:
                    self._switch_training_strategy()
                
                self.episode_count += 1
                
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.is_training = False
    
    def _train_rainbow_episode(self, curriculum_config: Dict) -> Dict:
        """Train Rainbow DQN agent for one episode"""
        
        # Reset game with curriculum settings
        state = self._reset_game_with_curriculum(curriculum_config)
        
        episode_reward = 0
        episode_steps = 0
        episode_losses = []
        
        while not self.game.game_state.game_over:
            
            # Get action from Rainbow agent
            compact_state = self.preprocessor.preprocess_compact_state(
                self.game.get_compact_state()
            )
            
            action = self.rainbow_agent.get_action(compact_state, exploration=True)
            
            # Execute action
            next_state_raw, reward, done, info = self.game.step(action)
            
            # Shape reward
            shaped_reward = self.reward_shaper.shape_reward(
                reward, self.game.game_state, info, curriculum_config
            )
            
            # Preprocess next state
            next_state = self.preprocessor.preprocess_compact_state(
                self.game.get_compact_state()
            )
            
            # Store experience
            self.rainbow_agent.step(
                compact_state, action, shaped_reward, next_state, done
            )
            
            episode_reward += shaped_reward
            episode_steps += 1
            self.total_steps += 1
            
            # Track losses
            if hasattr(self.rainbow_agent, 'loss_history') and self.rainbow_agent.loss_history:
                episode_losses.append(self.rainbow_agent.loss_history[-1])
            
            if done:
                break
        
        return {
            'agent': 'rainbow',
            'score': self.game.game_state.score,
            'reward': episode_reward,
            'steps': episode_steps,
            'losses': episode_losses,
            'success': self.game.game_state.score > 0,
            'curriculum_stage': curriculum_config['stage_name']
        }
    
    def _train_mcts_episode(self, curriculum_config: Dict) -> Dict:
        """Train using MCTS agent for one episode"""
        
        state = self._reset_game_with_curriculum(curriculum_config)
        
        episode_reward = 0
        episode_steps = 0
        search_times = []
        
        while not self.game.game_state.game_over:
            
            # Get action from MCTS
            action, mcts_info = self.mcts_agent.get_action(self.game.game_state)
            search_times.append(mcts_info['search_time'])
            
            # Execute action
            next_state_raw, reward, done, info = self.game.step(action)
            
            # Shape reward
            shaped_reward = self.reward_shaper.shape_reward(
                reward, self.game.game_state, info, curriculum_config
            )
            
            episode_reward += shaped_reward
            episode_steps += 1
            self.total_steps += 1
            
            if done:
                break
        
        return {
            'agent': 'mcts',
            'score': self.game.game_state.score,
            'reward': episode_reward,
            'steps': episode_steps,
            'search_times': search_times,
            'success': self.game.game_state.score > 0,
            'curriculum_stage': curriculum_config['stage_name']
        }
    
    def _train_ensemble_episode(self, curriculum_config: Dict) -> Dict:
        """Train ensemble model for one episode"""
        
        state = self._reset_game_with_curriculum(curriculum_config)
        
        episode_reward = 0
        episode_steps = 0
        decision_info = []
        
        while not self.game.game_state.game_over:
            
            # Get states for ensemble
            compact_state = torch.FloatTensor(
                self.preprocessor.preprocess_compact_state(
                    self.game.get_compact_state()
                )
            ).unsqueeze(0).to(self.device)
            
            cnn_state = torch.FloatTensor(
                self.preprocessor.preprocess_cnn_state(
                    self.game.get_state()
                )
            ).unsqueeze(0).to(self.device)
            
            # Get action from ensemble
            action, ensemble_results = self.ensemble_agent.get_best_action(
                compact_state, cnn_state, strategy='meta'
            )
            
            decision_info.append(ensemble_results)
            
            # Execute action
            next_state_raw, reward, done, info = self.game.step(action)
            
            # Shape reward
            shaped_reward = self.reward_shaper.shape_reward(
                reward, self.game.game_state, info, curriculum_config
            )
            
            episode_reward += shaped_reward
            episode_steps += 1
            self.total_steps += 1
            
            if done:
                break
        
        return {
            'agent': 'ensemble',
            'score': self.game.game_state.score,
            'reward': episode_reward,
            'steps': episode_steps,
            'decision_info': decision_info,
            'success': self.game.game_state.score > 0,
            'curriculum_stage': curriculum_config['stage_name']
        }
    
    def _train_competitive_episode(self, curriculum_config: Dict) -> Dict:
        """Train using competitive multi-agent setup"""
        
        results = []
        
        # Run multiple agents on same starting position
        initial_game_state = self._reset_game_with_curriculum(curriculum_config)
        
        agents = [
            ('rainbow', self.rainbow_agent),
            ('mcts', self.mcts_agent),
            ('ensemble', self.ensemble_agent)
        ]
        
        for agent_name, agent in agents:
            
            # Reset to same initial state
            self.game.game_state = initial_game_state
            
            episode_reward = 0
            episode_steps = 0
            
            while not self.game.game_state.game_over:
                
                if agent_name == 'rainbow':
                    compact_state = self.preprocessor.preprocess_compact_state(
                        self.game.get_compact_state()
                    )
                    action = agent.get_action(compact_state, exploration=True)
                    
                elif agent_name == 'mcts':
                    action, _ = agent.get_action(self.game.game_state)
                    
                elif agent_name == 'ensemble':
                    compact_state = torch.FloatTensor(
                        self.preprocessor.preprocess_compact_state(
                            self.game.get_compact_state()
                        )
                    ).unsqueeze(0).to(self.device)
                    
                    cnn_state = torch.FloatTensor(
                        self.preprocessor.preprocess_cnn_state(
                            self.game.get_state()
                        )
                    ).unsqueeze(0).to(self.device)
                    
                    action, _ = agent.get_best_action(compact_state, cnn_state)
                
                # Execute action
                next_state_raw, reward, done, info = self.game.step(action)
                
                # Shape reward
                shaped_reward = self.reward_shaper.shape_reward(
                    reward, self.game.game_state, info, curriculum_config
                )
                
                episode_reward += shaped_reward
                episode_steps += 1
                
                if done:
                    break
            
            result = {
                'agent': agent_name,
                'score': self.game.game_state.score,
                'reward': episode_reward,
                'steps': episode_steps,
                'success': self.game.game_state.score > 0,
                'curriculum_stage': curriculum_config['stage_name']
            }
            
            results.append(result)
            
            # Update agent performance tracking
            self.agent_performance[agent_name]['scores'].append(self.game.game_state.score)
        
        # Determine winner
        best_score = max(result['score'] for result in results)
        winners = [result for result in results if result['score'] == best_score]
        
        for winner in winners:
            self.agent_performance[winner['agent']]['wins'] += 1
        
        # Return result of best performing agent
        return max(results, key=lambda x: x['score'])
    
    def _reset_game_with_curriculum(self, curriculum_config: Dict):
        """Reset game with curriculum settings"""
        
        # Create new game state with curriculum configuration
        self.game.game_state = self.curriculum.create_game_environment()
        
        return self.game.get_state()
    
    def _update_curriculum(self, episode_result: Dict):
        """Update curriculum based on episode result"""
        
        self.curriculum.report_episode_result(
            score=episode_result['score'],
            steps=episode_result['steps'],
            success=episode_result['success']
        )
    
    def _update_performance_metrics(self, episode_result: Dict):
        """Update performance tracking metrics"""
        
        score = episode_result['score']
        self.recent_scores.append(score)
        
        if score > self.best_score:
            self.best_score = score
            self.logger.info(f"New best score: {self.best_score}")
        
        # Update metrics
        self.performance_metrics['scores'].append(score)
        self.performance_metrics['episode_lengths'].append(episode_result['steps'])
        
        if 'losses' in episode_result:
            self.performance_metrics['losses'].extend(episode_result['losses'])
        
        # Get curriculum progress
        curriculum_stats = self.curriculum.get_curriculum_statistics()
        self.performance_metrics['curriculum_progress'].append(curriculum_stats)
        
        # Log episode result
        if hasattr(self.rainbow_agent, 'get_statistics'):
            agent_stats = self.rainbow_agent.get_statistics()
            
            self.training_logger.log_episode(
                episode=self.episode_count,
                score=score,
                epsilon=agent_stats.get('epsilon', 0),
                loss=np.mean(episode_result.get('losses', [0])),
                q_value_mean=agent_stats.get('avg_q_value', 0),
                steps=episode_result['steps'],
                reward_total=episode_result['reward']
            )
    
    def _save_models(self):
        """Save all model checkpoints"""
        
        save_dir = self.config.MODEL_SAVE_PATH
        timestamp = int(time.time())
        
        # Save Rainbow agent
        rainbow_path = os.path.join(save_dir, f"rainbow_agent_{timestamp}.pth")
        self.rainbow_agent.save_models(rainbow_path)
        
        # Save ensemble model
        ensemble_path = os.path.join(save_dir, f"ensemble_model_{timestamp}.pth")
        torch.save({
            'model_state_dict': self.ensemble_agent.state_dict(),
            'episode_count': self.episode_count,
            'best_score': self.best_score,
            'performance_metrics': self.performance_metrics
        }, ensemble_path)
        
        # Save training statistics
        stats_path = os.path.join(save_dir, f"training_stats_{timestamp}.json")
        stats = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_score': self.best_score,
            'recent_average_score': np.mean(self.recent_scores) if self.recent_scores else 0,
            'curriculum_stats': self.curriculum.get_curriculum_statistics(),
            'agent_performance': self.agent_performance,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info(f"Models saved at episode {self.episode_count}")
    
    def _evaluate_models(self):
        """Evaluate all models on test scenarios"""
        
        self.logger.info("Starting model evaluation...")
        
        evaluation_results = {}
        test_episodes = 10
        
        # Test each agent
        for agent_name in ['rainbow', 'mcts', 'ensemble']:
            
            scores = []
            survival_times = []
            
            for _ in range(test_episodes):
                
                # Reset to evaluation environment
                self.game.reset()
                
                episode_steps = 0
                
                while not self.game.game_state.game_over and episode_steps < 1000:
                    
                    if agent_name == 'rainbow':
                        state = self.preprocessor.preprocess_compact_state(
                            self.game.get_compact_state()
                        )
                        action = self.rainbow_agent.get_action(state, exploration=False)
                        
                    elif agent_name == 'mcts':
                        action, _ = self.mcts_agent.get_action(self.game.game_state)
                        
                    elif agent_name == 'ensemble':
                        compact_state = torch.FloatTensor(
                            self.preprocessor.preprocess_compact_state(
                                self.game.get_compact_state()
                            )
                        ).unsqueeze(0).to(self.device)
                        
                        cnn_state = torch.FloatTensor(
                            self.preprocessor.preprocess_cnn_state(
                                self.game.get_state()
                            )
                        ).unsqueeze(0).to(self.device)
                        
                        action, _ = self.ensemble_agent.get_best_action(
                            compact_state, cnn_state, strategy='meta'
                        )
                    
                    _, _, done, _ = self.game.step(action)
                    episode_steps += 1
                    
                    if done:
                        break
                
                scores.append(self.game.game_state.score)
                survival_times.append(episode_steps)
            
            evaluation_results[agent_name] = {
                'average_score': np.mean(scores),
                'max_score': np.max(scores),
                'average_survival': np.mean(survival_times),
                'success_rate': np.mean([s > 0 for s in scores])
            }
        
        # Log evaluation results
        self.logger.info("Evaluation Results:")
        for agent_name, results in evaluation_results.items():
            self.logger.info(
                f"{agent_name}: Avg Score={results['average_score']:.2f}, "
                f"Max Score={results['max_score']}, "
                f"Avg Survival={results['average_survival']:.1f}, "
                f"Success Rate={results['success_rate']:.2%}"
            )
        
        return evaluation_results
    
    def _switch_training_strategy(self):
        """Switch between different training strategies"""
        
        current_idx = self.training_strategies.index(self.current_strategy)
        next_idx = (current_idx + 1) % len(self.training_strategies)
        self.current_strategy = self.training_strategies[next_idx]
        
        self.logger.info(f"Switched to training strategy: {self.current_strategy}")
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        
        status = {
            'is_training': self.is_training,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_score': self.best_score,
            'recent_average_score': np.mean(self.recent_scores) if self.recent_scores else 0,
            'current_strategy': self.current_strategy,
            'curriculum_stage': self.curriculum.current_stage.name,
            'curriculum_progress': self.curriculum.get_curriculum_statistics(),
            'agent_performance': self.agent_performance
        }
        
        if self.training_start_time:
            status['training_time'] = time.time() - self.training_start_time
        
        return status
    
    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics"""
        return self.performance_metrics.copy()
    
    def load_models(self, model_dir: str):
        """Load saved models"""
        
        try:
            # Load Rainbow agent
            rainbow_files = [f for f in os.listdir(model_dir) if f.startswith('rainbow_agent_')]
            if rainbow_files:
                latest_rainbow = max(rainbow_files)
                rainbow_path = os.path.join(model_dir, latest_rainbow)
                self.rainbow_agent.load_models(rainbow_path)
                self.logger.info(f"Loaded Rainbow agent from {rainbow_path}")
            
            # Load ensemble model
            ensemble_files = [f for f in os.listdir(model_dir) if f.startswith('ensemble_model_')]
            if ensemble_files:
                latest_ensemble = max(ensemble_files)
                ensemble_path = os.path.join(model_dir, latest_ensemble)
                
                checkpoint = torch.load(ensemble_path, map_location=self.device)
                self.ensemble_agent.load_state_dict(checkpoint['model_state_dict'])
                
                if 'episode_count' in checkpoint:
                    self.episode_count = checkpoint['episode_count']
                if 'best_score' in checkpoint:
                    self.best_score = checkpoint['best_score']
                
                self.logger.info(f"Loaded ensemble model from {ensemble_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def get_ai_analysis(self, game_state) -> Dict:
        """Get AI analysis for current game state"""
        
        analysis = {}
        
        # Rainbow agent analysis
        if hasattr(self.rainbow_agent, 'get_action_probabilities'):
            compact_state = self.preprocessor.preprocess_compact_state(
                game_state.get_compact_state()
            )
            analysis['rainbow'] = {
                'action_probabilities': self.rainbow_agent.get_action_probabilities(compact_state),
                'statistics': self.rainbow_agent.get_statistics()
            }
        
        # MCTS analysis
        mcts_analysis = self.mcts_agent.analyze_position(game_state)
        analysis['mcts'] = mcts_analysis
        
        # Ensemble analysis
        try:
            compact_state = torch.FloatTensor(
                self.preprocessor.preprocess_compact_state(
                    game_state.get_compact_state()
                )
            ).unsqueeze(0).to(self.device)
            
            cnn_state = torch.FloatTensor(
                self.preprocessor.preprocess_cnn_state(
                    game_state.get_state_array()
                )
            ).unsqueeze(0).to(self.device)
            
            action, ensemble_results = self.ensemble_agent.get_best_action(
                compact_state, cnn_state, strategy='meta'
            )
            
            analysis['ensemble'] = {
                'recommended_action': action,
                'model_predictions': ensemble_results,
                'model_rankings': self.ensemble_agent.get_model_rankings()
            }
            
        except Exception as e:
            self.logger.error(f"Error in ensemble analysis: {e}")
            analysis['ensemble'] = {'error': str(e)}
        
        return analysis
