"""
Ensemble methods combining multiple AI approaches for Snake AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

from ai.models.dqn_models import DuelingDQN, RainbowDQN, DoubleDQN
from ai.models.cnn_lstm_attention import CNNLSTMAttentionDQN, VisionTransformerDQN

class BaseEnsembleMember(ABC):
    """Base class for ensemble members"""
    
    @abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Predict Q-values for given state"""
        pass
    
    @abstractmethod
    def get_confidence(self, state: torch.Tensor) -> float:
        """Get confidence score for prediction"""
        pass

class DQNEnsembleMember(BaseEnsembleMember):
    """DQN-based ensemble member"""
    
    def __init__(self, model: nn.Module, weight: float = 1.0):
        self.model = model
        self.weight = weight
        self.confidence_history = []
    
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(state)
    
    def get_confidence(self, state: torch.Tensor) -> float:
        """Calculate confidence based on Q-value variance"""
        with torch.no_grad():
            q_values = self.model(state)
            variance = torch.var(q_values, dim=1).mean().item()
            # Higher variance = lower confidence
            confidence = 1.0 / (1.0 + variance)
            self.confidence_history.append(confidence)
            return confidence

class WeightedEnsemble(nn.Module):
    """Weighted ensemble of multiple models"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(WeightedEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0] * len(models)
        
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked = torch.stack(predictions, dim=0)  # (num_models, batch, actions)
        
        # Apply softmax to weights
        normalized_weights = self.softmax(self.weights)
        
        # Weighted average
        weighted_pred = torch.sum(
            stacked * normalized_weights.view(-1, 1, 1), 
            dim=0
        )
        
        return weighted_pred
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from individual models"""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        return predictions
    
    def update_weights(self, performance_scores: List[float]):
        """Update ensemble weights based on performance"""
        with torch.no_grad():
            performance_tensor = torch.tensor(performance_scores, dtype=torch.float32)
            # Normalize performance scores
            normalized_scores = F.softmax(performance_tensor, dim=0)
            # Update weights with momentum
            momentum = 0.9
            self.weights.data = momentum * self.weights.data + (1 - momentum) * normalized_scores

class AdaptiveEnsemble(nn.Module):
    """Adaptive ensemble that dynamically weights models based on context"""
    
    def __init__(self, models: List[nn.Module], input_size: int):
        super(AdaptiveEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        # Context-aware weighting network
        self.weight_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_models),
            nn.Softmax(dim=1)
        )
        
        # Performance tracking
        self.model_scores = torch.zeros(self.num_models)
        self.update_count = 0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Get context-dependent weights
        context_weights = self.weight_network(x.view(batch_size, -1))
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked = torch.stack(predictions, dim=1)  # (batch, num_models, actions)
        
        # Apply context weights
        weighted_pred = torch.sum(
            stacked * context_weights.unsqueeze(-1), 
            dim=1
        )
        
        return weighted_pred, context_weights
    
    def update_performance(self, model_idx: int, score: float):
        """Update performance tracking for a specific model"""
        self.model_scores[model_idx] = (
            self.model_scores[model_idx] * self.update_count + score
        ) / (self.update_count + 1)
        self.update_count += 1

class UncertaintyEnsemble(nn.Module):
    """Ensemble that uses uncertainty estimation for better decisions"""
    
    def __init__(self, models: List[nn.Module], num_samples: int = 10):
        super(UncertaintyEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_samples = num_samples
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation"""
        all_predictions = []
        
        for model in self.models:
            model_predictions = []
            
            # Enable dropout for uncertainty estimation
            model.train()
            
            for _ in range(self.num_samples):
                pred = model(x)
                model_predictions.append(pred)
            
            # Restore eval mode
            model.eval()
            
            # Stack samples
            model_preds = torch.stack(model_predictions, dim=0)
            all_predictions.append(model_preds)
        
        # Combine all model predictions
        combined_preds = torch.cat(all_predictions, dim=0)
        
        # Calculate mean and uncertainty
        mean_pred = torch.mean(combined_preds, dim=0)
        uncertainty = torch.std(combined_preds, dim=0)
        
        return mean_pred, uncertainty
    
    def get_epistemic_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Get epistemic uncertainty (model uncertainty)"""
        model_means = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                model_means.append(pred)
        
        model_means = torch.stack(model_means, dim=0)
        epistemic_uncertainty = torch.std(model_means, dim=0)
        
        return epistemic_uncertainty
    
    def get_aleatoric_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Get aleatoric uncertainty (data uncertainty)"""
        _, total_uncertainty = self.forward(x)
        epistemic_uncertainty = self.get_epistemic_uncertainty(x)
        
        # Aleatoric = Total - Epistemic
        aleatoric_uncertainty = torch.sqrt(
            torch.clamp(total_uncertainty**2 - epistemic_uncertainty**2, min=0)
        )
        
        return aleatoric_uncertainty

class HybridEnsemble(nn.Module):
    """Hybrid ensemble combining different architectures"""
    
    def __init__(self, compact_input_size: int, cnn_input_shape: Tuple[int, int, int]):
        super(HybridEnsemble, self).__init__()
        
        # Different model architectures
        self.dqn_model = DuelingDQN(compact_input_size)
        self.rainbow_model = RainbowDQN(compact_input_size)
        self.cnn_lstm_model = CNNLSTMAttentionDQN(cnn_input_shape)
        self.vit_model = VisionTransformerDQN(cnn_input_shape)
        
        # Ensemble combination
        self.weighted_ensemble = WeightedEnsemble([
            self.dqn_model, 
            self.rainbow_model
        ])
        
        # Meta-learning network for final decision
        self.meta_network = nn.Sequential(
            nn.Linear(16, 64),  # 4 models * 4 actions
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        
        # Performance tracking
        self.model_performance = {
            'dqn': [],
            'rainbow': [],
            'cnn_lstm': [],
            'vit': []
        }
    
    def forward(self, compact_state: torch.Tensor, 
                cnn_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all models"""
        
        results = {}
        
        # Compact state models
        dqn_pred = self.dqn_model(compact_state)
        results['dqn'] = dqn_pred
        
        # Rainbow model (returns q_values and distribution)
        if hasattr(self.rainbow_model, 'forward'):
            rainbow_output = self.rainbow_model(compact_state)
            if isinstance(rainbow_output, tuple):
                rainbow_pred, _ = rainbow_output
            else:
                rainbow_pred = rainbow_output
        else:
            rainbow_pred = dqn_pred  # Fallback
        results['rainbow'] = rainbow_pred
        
        # CNN-based models
        cnn_lstm_pred = self.cnn_lstm_model(cnn_state)
        results['cnn_lstm'] = cnn_lstm_pred
        
        vit_pred = self.vit_model(cnn_state)
        results['vit'] = vit_pred
        
        # Weighted ensemble of simple models
        ensemble_pred = self.weighted_ensemble(compact_state)
        results['ensemble'] = ensemble_pred
        
        # Meta-learning combination
        all_preds = torch.cat([
            dqn_pred, rainbow_pred, cnn_lstm_pred, vit_pred
        ], dim=1)
        
        meta_pred = self.meta_network(all_preds)
        results['meta'] = meta_pred
        
        return results
    
    def get_best_action(self, compact_state: torch.Tensor, 
                       cnn_state: torch.Tensor,
                       strategy: str = 'meta') -> Tuple[int, Dict]:
        """Get best action using specified strategy"""
        
        with torch.no_grad():
            results = self.forward(compact_state, cnn_state)
            
            if strategy == 'meta':
                q_values = results['meta']
            elif strategy == 'ensemble':
                q_values = results['ensemble']
            elif strategy == 'voting':
                # Majority voting
                actions = []
                for key in ['dqn', 'rainbow', 'cnn_lstm', 'vit']:
                    action = torch.argmax(results[key], dim=1)
                    actions.append(action)
                
                # Get most common action
                action_counts = torch.bincount(torch.cat(actions))
                best_action = torch.argmax(action_counts).item()
                
                return best_action, results
            else:
                q_values = results.get(strategy, results['meta'])
            
            best_action = torch.argmax(q_values, dim=1).item()
            
            return best_action, results
    
    def update_performance(self, model_name: str, score: float):
        """Update performance tracking"""
        if model_name in self.model_performance:
            self.model_performance[model_name].append(score)
            
            # Keep only recent performance
            if len(self.model_performance[model_name]) > 100:
                self.model_performance[model_name] = self.model_performance[model_name][-100:]
    
    def get_model_rankings(self) -> Dict[str, float]:
        """Get current model performance rankings"""
        rankings = {}
        
        for model_name, scores in self.model_performance.items():
            if scores:
                rankings[model_name] = np.mean(scores[-20:])  # Recent average
            else:
                rankings[model_name] = 0.0
        
        return rankings

class DynamicEnsemble(nn.Module):
    """Dynamic ensemble that adapts model selection based on game state"""
    
    def __init__(self, models: Dict[str, nn.Module]):
        super(DynamicEnsemble, self).__init__()
        
        self.models = nn.ModuleDict(models)
        
        # State classifier to determine which model to use
        self.state_classifier = nn.Sequential(
            nn.Linear(11, 64),  # Compact state size
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(models)),
            nn.Softmax(dim=1)
        )
        
        # Performance per state type
        self.state_performance = {}
    
    def forward(self, compact_state: torch.Tensor, 
                cnn_state: torch.Tensor) -> torch.Tensor:
        """Dynamic forward pass"""
        
        # Classify state type
        state_weights = self.state_classifier(compact_state)
        
        # Get predictions from relevant models
        predictions = []
        model_names = list(self.models.keys())
        
        for model_name in model_names:
            model = self.models[model_name]
            
            if 'cnn' in model_name.lower() or 'vit' in model_name.lower():
                pred = model(cnn_state)
            else:
                pred = model(compact_state)
            
            predictions.append(pred)
        
        # Weighted combination
        stacked = torch.stack(predictions, dim=0)
        weights = state_weights.T.unsqueeze(-1)
        
        result = torch.sum(stacked * weights, dim=0)
        
        return result
    
    def train_state_classifier(self, states: List[torch.Tensor], 
                             performances: List[Dict[str, float]]):
        """Train the state classifier based on model performance"""
        
        optimizer = torch.optim.Adam(self.state_classifier.parameters(), lr=0.001)
        
        for state, perf_dict in zip(states, performances):
            # Create target distribution based on performance
            model_names = list(self.models.keys())
            target = torch.tensor([perf_dict.get(name, 0.0) for name in model_names])
            target = F.softmax(target, dim=0).unsqueeze(0)
            
            # Forward pass
            pred_weights = self.state_classifier(state)
            
            # Loss
            loss = F.kl_div(torch.log(pred_weights), target, reduction='batchmean')
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
