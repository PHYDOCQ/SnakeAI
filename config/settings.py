"""
Configuration settings for the Advanced Snake AI System
"""

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class Config:
    """Configuration class for the Snake AI system"""
    
    # Web server settings
    WEB_PORT: int = 5000
    DEBUG: bool = True
    
    # Game settings
    GRID_SIZE: int = 20
    GAME_WIDTH: int = 600
    GAME_HEIGHT: int = 600
    FPS: int = 10
    
    # AI training settings
    LEARNING_RATE: float = 0.0001
    BATCH_SIZE: int = 64
    MEMORY_SIZE: int = 100000
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY: float = 0.995
    GAMMA: float = 0.99
    TARGET_UPDATE: int = 1000
    
    # Model architecture settings
    CNN_CHANNELS: Tuple[int, ...] = (32, 64, 128)
    LSTM_HIDDEN_SIZE: int = 256
    ATTENTION_HEADS: int = 8
    DROPOUT_RATE: float = 0.1
    
    # Advanced training settings
    PRIORITIZED_REPLAY_ALPHA: float = 0.6
    PRIORITIZED_REPLAY_BETA: float = 0.4
    MULTI_STEP_N: int = 3
    RAINBOW_ATOMS: int = 51
    V_MIN: float = -10.0
    V_MAX: float = 10.0
    
    # MCTS settings
    MCTS_SIMULATIONS: int = 100
    MCTS_C_PUCT: float = 1.0
    
    # Curriculum learning
    CURRICULUM_STAGES: int = 5
    STAGE_EPISODES: int = 1000
    
    # File paths
    MODEL_SAVE_PATH: str = "models"
    LOG_PATH: str = "logs"
    DATA_PATH: str = "data"
    
    # Training control
    AUTO_START_TRAINING: bool = True
    SAVE_INTERVAL: int = 100
    EVALUATION_INTERVAL: int = 50
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.LOG_PATH, exist_ok=True)
        os.makedirs(self.DATA_PATH, exist_ok=True)
    
    @property
    def device(self) -> str:
        """Get the appropriate device for PyTorch"""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
