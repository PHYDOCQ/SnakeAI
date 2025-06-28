"""
Rainbow DQN agent with all improvements integrated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
import random
from collections import deque

from ai.models.dqn_models import RainbowDQN, NoisyLinear
from ai.training.experience_replay import PrioritizedReplayBuffer
from ai.utils.preprocessing import StatePreprocessor
from utils.logger import setup_logger

class RainbowAgent:
    """Advanced Rainbow DQN agent with all improvements"""
    
    def __init__(self, state_size: int, action_size: int = 4, config=None):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device(config.device if config else "cpu")
        self.logger = setup_logger('rainbow_agent')
        
        # Rainbow DQN specific parameters
        self.atoms = config.RAINBOW_ATOMS if config else 51
        self.v_min = config.V_MIN if config else -10.0
        self.v_max = config.V_MAX if config else 10.0
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(self.device)
        
        # Networks
        self.q_network = RainbowDQN(
            state_size, 
            atoms=self.atoms,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)
        
        self.target_network = RainbowDQN(
            state_size,
            atoms=self.atoms,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.LEARNING_RATE if config else 0.0001
        )
        
        # Experience replay
        memory_size = config.MEMORY_SIZE if config else 100000
        self.memory = PrioritizedReplayBuffer(
            memory_size,
            alpha=config.PRIORITIZED_REPLAY_ALPHA if config else 0.6
        )
        
        # Multi-step learning
        self.n_step = config.MULTI_STEP_N if config else 3
        self.gamma = config.GAMMA if config else 0.99
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # Training parameters
        self.batch_size = config.BATCH_SIZE if config else 64
        self.target_update_freq = config.TARGET_UPDATE if config else 1000
        self.beta_start = config.PRIORITIZED_REPLAY_BETA if config else 0.4
        self.beta_frames = 100000
        
        # Counters
        self.frame_count = 0
        self.episode_count = 0
        
        # State preprocessor
        self.preprocessor = StatePreprocessor()
        
        # Performance tracking
        self.q_values_history = []
        self.loss_history = []
        
    def get_action(self, state: np.ndarray, exploration: bool = True) -> int:
        """Get action using noisy networks (no epsilon needed)"""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Reset noise in noisy layers
        if exploration:
            self.q_network.reset_noise()
        
        with torch.no_grad():
            q_values, _ = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        # Track Q-values for analysis
        self.q_values_history.append(q_values.cpu().numpy().flatten())
        
        return action
    
    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Process one step of experience"""
        
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If buffer is full, calculate n-step return and add to memory
        if len(self.n_step_buffer) == self.n_step:
            state_0, action_0, _, _, _ = self.n_step_buffer[0]
            
            # Calculate n-step return
            n_step_return = 0
            gamma_power = 1
            
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_return += gamma_power * r
                gamma_power *= self.gamma
                
                if d:  # Episode ended
                    break
            
            # Get final state and done flag
            _, _, _, next_state_n, done_n = self.n_step_buffer[-1]
            
            # Add to prioritized replay buffer
            self.memory.add(state_0, action_0, n_step_return, next_state_n, done_n)
        
        # Learn from experience
        if len(self.memory) > self.batch_size:
            self.learn()
        
        # Update target network
        if self.frame_count % self.target_update_freq == 0:
            self.update_target_network()
        
        self.frame_count += 1
    
    def learn(self):
        """Learn from a batch of experiences"""
        
        # Calculate beta for importance sampling
        beta = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        
        # Sample from prioritized replay buffer
        experiences, indices, weights = self.memory.sample(self.batch_size, beta)
        
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values and distributions
        current_q_values, current_dist = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Get current distribution for selected actions
        current_dist = current_dist[range(self.batch_size), actions]
        
        # Next Q-values for Double DQN
        with torch.no_grad():
            next_q_values, _ = self.q_network(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            _, next_dist = self.target_network(next_states)
            next_dist = next_dist[range(self.batch_size), next_actions]
            
            # Compute target distribution
            target_dist = self._compute_target_distribution(
                rewards, next_dist, dones
            )
        
        # Compute loss (Cross-entropy between distributions)
        loss = -torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=1)
        
        # Apply importance sampling weights
        weighted_loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        
        self.optimizer.step()
        
        # Update priorities
        td_errors = loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Track loss
        self.loss_history.append(weighted_loss.item())
        
    def _compute_target_distribution(self, rewards: torch.Tensor, 
                                   next_dist: torch.Tensor, 
                                   dones: torch.Tensor) -> torch.Tensor:
        """Compute target distribution for distributional RL"""
        
        batch_size = rewards.size(0)
        delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        
        target_dist = torch.zeros_like(next_dist)
        
        for i in range(batch_size):
            if dones[i]:
                # Terminal state
                tz = rewards[i].clamp(self.v_min, self.v_max)
                b = (tz - self.v_min) / delta_z
                l, u = b.floor().long(), b.ceil().long()
                target_dist[i, l] = 1.0
            else:
                # Non-terminal state
                for j in range(self.atoms):
                    tz = rewards[i] + self.gamma**self.n_step * self.support[j]
                    tz = tz.clamp(self.v_min, self.v_max)
                    b = (tz - self.v_min) / delta_z
                    l, u = b.floor().long(), b.ceil().long()
                    
                    target_dist[i, l] += next_dist[i, j] * (u - b)
                    target_dist[i, u] += next_dist[i, j] * (b - l)
        
        return target_dist
    
    def update_target_network(self):
        """Update target network with soft update"""
        tau = 0.005
        for target_param, main_param in zip(
            self.target_network.parameters(), 
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * main_param.data + (1.0 - tau) * target_param.data
            )
    
    def save_models(self, filepath: str):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frame_count': self.frame_count,
            'episode_count': self.episode_count
        }, filepath)
        
        self.logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.frame_count = checkpoint.get('frame_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        self.logger.info(f"Models loaded from {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        recent_q_values = self.q_values_history[-100:] if self.q_values_history else []
        recent_losses = self.loss_history[-100:] if self.loss_history else []
        
        stats = {
            'frame_count': self.frame_count,
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'avg_q_value': np.mean(recent_q_values) if recent_q_values else 0,
            'max_q_value': np.max(recent_q_values) if recent_q_values else 0,
            'avg_loss': np.mean(recent_losses) if recent_losses else 0,
            'beta': min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        }
        
        return stats
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for visualization"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, _ = self.q_network(state_tensor)
            probabilities = F.softmax(q_values / 0.1, dim=1)  # Temperature scaling
            
        return probabilities.cpu().numpy().flatten()
    
    def reset_noise(self):
        """Reset noise in noisy networks"""
        self.q_network.reset_noise()
        self.target_network.reset_noise()
    
    def set_eval_mode(self):
        """Set networks to evaluation mode"""
        self.q_network.eval()
        self.target_network.eval()
    
    def set_train_mode(self):
        """Set networks to training mode"""
        self.q_network.train()
        self.target_network.eval()  # Target network always in eval mode
