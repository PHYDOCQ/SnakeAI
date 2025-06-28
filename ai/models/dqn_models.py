"""
Deep Q-Network models with advanced architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

class DuelingDQN(nn.Module):
    """Dueling DQN architecture"""
    
    def __init__(self, input_size: int, hidden_size: int = 512, output_size: int = 4):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class NoisyLinear(nn.Module):
    """Noisy linear layer for NoisyNet"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)

class RainbowDQN(nn.Module):
    """Rainbow DQN with distributional RL"""
    
    def __init__(self, input_size: int, hidden_size: int = 512, 
                 output_size: int = 4, atoms: int = 51, 
                 v_min: float = -10, v_max: float = 10):
        super(RainbowDQN, self).__init__()
        
        self.output_size = output_size
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Support for distributional RL
        self.register_buffer('support', torch.linspace(v_min, v_max, atoms))
        
        # Feature layers with noisy networks
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Dueling architecture with noisy layers
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            NoisyLinear(hidden_size // 2, atoms)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            NoisyLinear(hidden_size // 2, output_size * atoms)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = self.feature_layer(x)
        
        value = self.value_stream(features).view(batch_size, 1, self.atoms)
        advantage = self.advantage_stream(features).view(
            batch_size, self.output_size, self.atoms
        )
        
        # Combine value and advantage
        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Apply softmax to get probability distributions
        dist = F.softmax(q_atoms, dim=2)
        
        # Compute Q-values as expected values
        q_values = torch.sum(dist * self.support, dim=2)
        
        return q_values, dist
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class DoubleDQN(nn.Module):
    """Double DQN implementation"""
    
    def __init__(self, input_size: int, hidden_size: int = 512, output_size: int = 4):
        super(DoubleDQN, self).__init__()
        
        self.online_net = DuelingDQN(input_size, hidden_size, output_size)
        self.target_net = DuelingDQN(input_size, hidden_size, output_size)
        
        # Initialize target network with same weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Freeze target network
        for param in self.target_net.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        if use_target:
            return self.target_net(x)
        return self.online_net(x)
    
    def update_target_network(self, tau: float = 0.005):
        """Soft update of target network"""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

class MultiStepDQN(nn.Module):
    """Multi-step DQN for n-step learning"""
    
    def __init__(self, input_size: int, hidden_size: int = 512, 
                 output_size: int = 4, n_step: int = 3):
        super(MultiStepDQN, self).__init__()
        
        self.n_step = n_step
        self.dqn = DuelingDQN(input_size, hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dqn(x)
    
    def compute_n_step_targets(self, rewards: torch.Tensor, 
                              next_q_values: torch.Tensor,
                              dones: torch.Tensor, gamma: float) -> torch.Tensor:
        """Compute n-step targets"""
        batch_size = rewards.size(0)
        targets = torch.zeros_like(next_q_values)
        
        for i in range(batch_size):
            n_step_reward = 0
            gamma_power = 1
            
            for step in range(self.n_step):
                if i + step < batch_size and not dones[i + step]:
                    n_step_reward += gamma_power * rewards[i + step]
                    gamma_power *= gamma
                else:
                    break
            
            if i + self.n_step < batch_size and not dones[i + self.n_step]:
                targets[i] = n_step_reward + gamma_power * next_q_values[i + self.n_step]
            else:
                targets[i] = n_step_reward
        
        return targets
