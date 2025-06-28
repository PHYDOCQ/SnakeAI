#!/usr/bin/env python3
"""
Advanced Snake AI System
Main entry point for the sophisticated Snake AI using PyTorch
"""

import sys
import os
import threading
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from web.app import create_app
from ai.training.trainer import AdvancedTrainer
from game.snake_game import SnakeGame
from config.settings import Config
from utils.logger import setup_logger

def main():
    """Main entry point for the Snake AI system"""
    
    # Setup logging
    logger = setup_logger('main')
    logger.info("Starting Advanced Snake AI System")
    
    # Initialize configuration
    config = Config()
    
    try:
        # Create Flask web application
        app = create_app(config)
        
        # Start training in background thread
        if config.AUTO_START_TRAINING:
            trainer = AdvancedTrainer(config)
            training_thread = threading.Thread(
                target=trainer.start_training,
                daemon=True
            )
            training_thread.start()
            logger.info("Background training started")
        
        # Start web server
        logger.info(f"Starting web server on http://0.0.0.0:{config.WEB_PORT}")
        app.run(
            host='0.0.0.0',
            port=config.WEB_PORT,
            debug=config.DEBUG,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down Snake AI System")
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
