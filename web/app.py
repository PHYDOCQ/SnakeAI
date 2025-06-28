"""
Flask web application for Snake AI visualization and control
"""

from flask import Flask, render_template, jsonify, request, send_file
import json
import threading
import time
import base64
import io
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional

from game.snake_game import SnakeGame
from ai.training.trainer import AdvancedTrainer
from visualization.game_visualizer import GameVisualizer, TrainingVisualizer
from visualization.ai_analyzer import AIAnalyzer
from config.settings import Config
from utils.logger import setup_logger

def create_app(config: Config) -> Flask:
    """Create and configure Flask application"""
    
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY='snake_ai_secret_key_12345',
        DEBUG=config.DEBUG
    )
    
    # Initialize components
    logger = setup_logger('web_app')
    game_visualizer = GameVisualizer(800, 600, config.GRID_SIZE)
    training_visualizer = TrainingVisualizer()
    ai_analyzer = AIAnalyzer()
    
    # Initialize game and trainer
    trainer = AdvancedTrainer(config)
    game = SnakeGame(config, headless=True)
    
    # Web app state
    app_state = {
        'game_running': False,
        'training_active': False,
        'demo_mode': False,
        'visualization_mode': 'all',
        'current_agent': 'rainbow',
        'last_update': time.time()
    }
    
    @app.route('/')
    def index():
        """Main page"""
        return render_template('index.html')
    
    @app.route('/api/status')
    def get_status():
        """Get current system status"""
        try:
            training_status = trainer.get_training_status()
            
            status = {
                'system_online': True,
                'training_active': training_status['is_training'],
                'current_agent': app_state['current_agent'],
                'game_running': app_state['game_running'],
                'demo_mode': app_state['demo_mode'],
                'episode_count': training_status['episode_count'],
                'best_score': training_status['best_score'],
                'recent_average': training_status['recent_average_score'],
                'curriculum_stage': training_status['curriculum_stage'],
                'training_time': training_status.get('training_time', 0),
                'visualization_mode': app_state['visualization_mode']
            }
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/training/start', methods=['POST'])
    def start_training():
        """Start training process"""
        try:
            if not trainer.is_training:
                trainer.start_training()
                app_state['training_active'] = True
                
                return jsonify({
                    'success': True,
                    'message': 'Training started successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Training is already active'
                })
                
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/training/stop', methods=['POST'])
    def stop_training():
        """Stop training process"""
        try:
            if trainer.is_training:
                trainer.stop_training_process()
                app_state['training_active'] = False
                
                return jsonify({
                    'success': True,
                    'message': 'Training stopped successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Training is not active'
                })
                
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/game/start', methods=['POST'])
    def start_game():
        """Start game demo"""
        try:
            data = request.get_json() or {}
            agent_type = data.get('agent', 'rainbow')
            
            app_state['current_agent'] = agent_type
            app_state['game_running'] = True
            app_state['demo_mode'] = True
            
            # Reset game
            game.reset()
            
            return jsonify({
                'success': True,
                'message': f'Game started with {agent_type} agent'
            })
            
        except Exception as e:
            logger.error(f"Error starting game: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/game/stop', methods=['POST'])
    def stop_game():
        """Stop game demo"""
        try:
            app_state['game_running'] = False
            app_state['demo_mode'] = False
            
            return jsonify({
                'success': True,
                'message': 'Game stopped'
            })
            
        except Exception as e:
            logger.error(f"Error stopping game: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/game/step', methods=['POST'])
    def game_step():
        """Execute one game step"""
        try:
            if not app_state['game_running']:
                return jsonify({'error': 'Game not running'}), 400
            
            # Get AI decision and analysis
            ai_analysis = trainer.get_ai_analysis(game.game_state)
            
            # Get action from selected agent
            action = 0  # Default action
            
            if app_state['current_agent'] == 'rainbow':
                if 'rainbow' in ai_analysis:
                    action_probs = ai_analysis['rainbow'].get('action_probabilities', [0.25] * 4)
                    action = np.argmax(action_probs)
            
            elif app_state['current_agent'] == 'mcts':
                if 'mcts' in ai_analysis:
                    # MCTS returns action directly
                    action = ai_analysis['mcts'].get('best_action', 0)
            
            elif app_state['current_agent'] == 'ensemble':
                if 'ensemble' in ai_analysis:
                    action = ai_analysis['ensemble'].get('recommended_action', 0)
            
            # Execute action
            next_state, reward, done, info = game.step(action)
            
            # Analyze the decision
            decision_analysis = ai_analyzer.analyze_decision_making(
                game.game_state, ai_analysis, action
            )
            
            # Generate explanation
            explanation = ai_analyzer.generate_decision_explanation(
                game.game_state, ai_analysis, action
            )
            
            # Prepare response
            response = {
                'game_state': {
                    'score': game.game_state.score,
                    'steps': game.game_state.steps,
                    'snake_length': len(game.game_state.snake),
                    'game_over': game.game_state.game_over,
                    'snake_positions': game.game_state.snake,
                    'food_position': game.game_state.food
                },
                'ai_analysis': {
                    'action_taken': action,
                    'action_name': ['UP', 'RIGHT', 'DOWN', 'LEFT'][action],
                    'explanation': explanation,
                    'confidence': decision_analysis['confidence_metrics']['decision_confidence'],
                    'decision_quality': decision_analysis['decision_quality']['overall_score'],
                    'strategy_type': decision_analysis['strategic_assessment']['strategy_type'],
                    'risk_level': decision_analysis['strategic_assessment']['risk_level']
                },
                'reward': reward,
                'done': done
            }
            
            # Add agent-specific analysis
            if app_state['current_agent'] in ai_analysis:
                agent_data = ai_analysis[app_state['current_agent']]
                if 'action_probabilities' in agent_data:
                    response['ai_analysis']['action_probabilities'] = agent_data['action_probabilities']
            
            # Reset game if done
            if done and app_state['demo_mode']:
                game.reset()
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in game step: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/visualization/image')
    def get_visualization_image():
        """Get current game visualization as image"""
        try:
            if not app_state['game_running']:
                return jsonify({'error': 'Game not running'}), 400
            
            # Get AI analysis for visualization
            ai_analysis = trainer.get_ai_analysis(game.game_state)
            
            # Create visualization overlay data
            visualization_data = {
                'attention_map': np.random.random((config.GRID_SIZE, config.GRID_SIZE)),  # Placeholder
                'action_probabilities': ai_analysis.get(app_state['current_agent'], {}).get('action_probabilities', [0.25] * 4),
                'q_values': [0.5, 0.3, 0.1, 0.4],  # Placeholder
                'predicted_path': [],  # Would come from planning algorithms
                'exploration_map': np.zeros((config.GRID_SIZE, config.GRID_SIZE))  # Placeholder
            }
            
            # Render game with AI overlay
            surface = game_visualizer.render_game_with_ai(game.game_state, visualization_data)
            
            # Convert to image
            frame = game_visualizer.create_video_frame()
            
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            return jsonify({
                'image': f"data:image/png;base64,{image_data}",
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/training/metrics')
    def get_training_metrics():
        """Get training metrics and statistics"""
        try:
            training_status = trainer.get_training_status()
            performance_metrics = trainer.get_performance_metrics()
            
            # Create training visualization
            metrics_image = training_visualizer.plot_training_metrics(performance_metrics)
            
            # Convert to base64
            image = Image.fromarray(metrics_image)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            metrics_image_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Get curriculum progress
            curriculum_stats = training_status.get('curriculum_progress', {})
            curriculum_image = training_visualizer.plot_curriculum_progress(curriculum_stats)
            
            # Convert to base64
            image = Image.fromarray(curriculum_image)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            curriculum_image_data = base64.b64encode(buffer.getvalue()).decode()
            
            response = {
                'training_status': training_status,
                'performance_metrics': {
                    'recent_scores': performance_metrics.get('scores', [])[-50:],
                    'recent_losses': performance_metrics.get('losses', [])[-100:],
                    'recent_q_values': performance_metrics.get('q_values', [])[-50:],
                    'episode_lengths': performance_metrics.get('episode_lengths', [])[-50:]
                },
                'visualizations': {
                    'training_metrics': f"data:image/png;base64,{metrics_image_data}",
                    'curriculum_progress': f"data:image/png;base64,{curriculum_image_data}"
                },
                'agent_performance': training_status.get('agent_performance', {}),
                'curriculum_info': curriculum_stats
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error getting training metrics: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/agents/compare')
    def compare_agents():
        """Compare performance of different AI agents"""
        try:
            training_status = trainer.get_training_status()
            agent_performance = training_status.get('agent_performance', {})
            
            comparison = {}
            
            for agent_name, performance in agent_performance.items():
                scores = performance.get('scores', [])
                if scores:
                    comparison[agent_name] = {
                        'average_score': np.mean(scores),
                        'max_score': max(scores),
                        'min_score': min(scores),
                        'consistency': 1.0 / (1.0 + np.std(scores)),  # Higher is more consistent
                        'total_wins': performance.get('wins', 0),
                        'games_played': len(scores),
                        'recent_performance': np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
                    }
                else:
                    comparison[agent_name] = {
                        'average_score': 0,
                        'max_score': 0,
                        'min_score': 0,
                        'consistency': 0,
                        'total_wins': 0,
                        'games_played': 0,
                        'recent_performance': 0
                    }
            
            return jsonify({
                'comparison': comparison,
                'recommendation': _get_agent_recommendation(comparison)
            })
            
        except Exception as e:
            logger.error(f"Error comparing agents: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/analysis/behavior')
    def get_behavior_analysis():
        """Get AI behavior analysis"""
        try:
            behavior_profile = ai_analyzer.create_behavior_profile()
            recent_performance = ai_analyzer._analyze_recent_performance()
            insights = ai_analyzer._generate_learning_insights()
            recommendations = ai_analyzer._generate_recommendations()
            
            analysis = {
                'behavior_profile': behavior_profile,
                'recent_performance': recent_performance,
                'insights': insights,
                'recommendations': recommendations,
                'decision_history_length': len(ai_analyzer.decision_history)
            }
            
            return jsonify(analysis)
            
        except Exception as e:
            logger.error(f"Error getting behavior analysis: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/config/visualization', methods=['POST'])
    def update_visualization_config():
        """Update visualization configuration"""
        try:
            data = request.get_json()
            
            if 'mode' in data:
                app_state['visualization_mode'] = data['mode']
            
            # Update game visualizer settings
            if 'show_attention' in data:
                game_visualizer.toggle_visualization_mode('attention')
            if 'show_decision_path' in data:
                game_visualizer.toggle_visualization_mode('decision_path')
            if 'show_safety_zones' in data:
                game_visualizer.toggle_visualization_mode('safety_zones')
            if 'show_q_values' in data:
                game_visualizer.toggle_visualization_mode('q_values')
            
            return jsonify({
                'success': True,
                'current_config': {
                    'visualization_mode': app_state['visualization_mode'],
                    'show_attention': game_visualizer.show_attention,
                    'show_decision_path': game_visualizer.show_decision_path,
                    'show_safety_zones': game_visualizer.show_safety_zones,
                    'show_q_values': game_visualizer.show_q_values
                }
            })
            
        except Exception as e:
            logger.error(f"Error updating visualization config: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/export/analysis')
    def export_analysis():
        """Export analysis report"""
        try:
            filename = f"snake_ai_analysis_{int(time.time())}.json"
            success = ai_analyzer.export_analysis_report(filename)
            
            if success:
                return send_file(filename, as_attachment=True)
            else:
                return jsonify({'error': 'Failed to generate report'}), 500
                
        except Exception as e:
            logger.error(f"Error exporting analysis: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _get_agent_recommendation(comparison: Dict) -> str:
        """Get recommendation for best agent based on comparison"""
        
        if not comparison:
            return "No data available for recommendation"
        
        # Score agents based on multiple criteria
        agent_scores = {}
        
        for agent, metrics in comparison.items():
            score = 0
            
            # Weight different factors
            score += metrics['average_score'] * 0.3
            score += metrics['max_score'] * 0.2
            score += metrics['consistency'] * 0.2
            score += metrics['recent_performance'] * 0.3
            
            agent_scores[agent] = score
        
        best_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
        
        return f"Recommended agent: {best_agent} based on overall performance"
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    # Cleanup on app shutdown
    @app.teardown_appcontext
    def cleanup(error):
        if hasattr(game_visualizer, 'cleanup'):
            game_visualizer.cleanup()
    
    logger.info("Flask application created successfully")
    
    return app

