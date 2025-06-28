"""
Logging utilities for the Snake AI system
"""

import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with file and console handlers
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'logs/{name}_{timestamp}.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

class TrainingLogger:
    """Specialized logger for training metrics"""
    
    def __init__(self, name: str = "training"):
        self.logger = setup_logger(name)
        self.metrics_file = f"logs/training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Initialize CSV file
        with open(self.metrics_file, 'w') as f:
            f.write("episode,score,epsilon,loss,q_value_mean,steps,reward_total\n")
    
    def log_episode(self, episode: int, score: int, epsilon: float, 
                   loss: float, q_value_mean: float, steps: int, reward_total: float):
        """Log episode metrics"""
        
        # Log to file
        self.logger.info(
            f"Episode {episode}: Score={score}, Steps={steps}, "
            f"Epsilon={epsilon:.4f}, Loss={loss:.4f}, Q-mean={q_value_mean:.4f}"
        )
        
        # Save to CSV
        with open(self.metrics_file, 'a') as f:
            f.write(f"{episode},{score},{epsilon},{loss},{q_value_mean},{steps},{reward_total}\n")
