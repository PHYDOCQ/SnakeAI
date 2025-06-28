"""
State preprocessing utilities for Snake AI
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Any
import torch
from collections import deque

from game.game_state import GameState
from utils.logger import setup_logger

class StatePreprocessor:
    """Advanced state preprocessing for Snake AI"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = setup_logger('state_preprocessor')
        
        # Normalization parameters
        self.state_mean = None
        self.state_std = None
        self.update_normalization = True
        
        # Frame stacking for temporal information
        self.frame_stack_size = 4
        self.frame_buffer = deque(maxlen=self.frame_stack_size)
        
        # State augmentation
        self.augment_states = True
        self.augmentation_prob = 0.1
        
        # Feature extraction parameters
        self.vision_distance = 8
        self.spatial_resolution = 32
        
    def preprocess_cnn_state(self, state: np.ndarray) -> np.ndarray:
        """
        Preprocess state for CNN input
        
        Args:
            state: Raw state array (height, width, channels)
            
        Returns:
            Preprocessed state ready for CNN
        """
        
        # Ensure proper dimensions
        if len(state.shape) == 2:
            # Add channel dimension
            state = np.expand_dims(state, axis=-1)
        
        # Resize to standard resolution
        if state.shape[:2] != (self.spatial_resolution, self.spatial_resolution):
            state = self._resize_state(state, (self.spatial_resolution, self.spatial_resolution))
        
        # Normalize to [0, 1]
        state = state.astype(np.float32)
        if state.max() > 1.0:
            state = state / 255.0
        
        # Apply data augmentation if enabled
        if self.augment_states and np.random.random() < self.augmentation_prob:
            state = self._augment_state(state)
        
        # Add frame stacking
        state = self._add_frame_stacking(state)
        
        # Convert to CHW format for PyTorch
        state = np.transpose(state, (2, 0, 1))
        
        return state
    
    def preprocess_compact_state(self, state: np.ndarray) -> np.ndarray:
        """
        Preprocess compact state representation
        
        Args:
            state: Compact state vector
            
        Returns:
            Preprocessed state vector
        """
        
        state = state.astype(np.float32)
        
        # Apply normalization
        if self.state_mean is not None and self.state_std is not None:
            state = (state - self.state_mean) / (self.state_std + 1e-8)
        elif self.update_normalization:
            self._update_normalization_params(state)
        
        # Clip extreme values
        state = np.clip(state, -5.0, 5.0)
        
        return state
    
    def preprocess_vision_state(self, game_state: GameState) -> np.ndarray:
        """
        Create vision-based state representation
        
        Args:
            game_state: Current game state
            
        Returns:
            Vision-based state vector
        """
        
        if not game_state.snake:
            return np.zeros((self.vision_distance * 8 * 3 + 10,), dtype=np.float32)
        
        head_x, head_y = game_state.snake[0]
        vision_vector = []
        
        # 8 directional rays
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Up-left, Up, Up-right
            (0, -1),           (0, 1),   # Left, Right
            (1, -1),  (1, 0),  (1, 1)    # Down-left, Down, Down-right
        ]
        
        for dx, dy in directions:
            ray_info = self._cast_vision_ray(
                game_state, head_x, head_y, dx, dy, self.vision_distance
            )
            vision_vector.extend(ray_info)
        
        # Add snake body information
        body_info = self._get_body_information(game_state)
        vision_vector.extend(body_info)
        
        # Add food information
        food_info = self._get_food_information(game_state)
        vision_vector.extend(food_info)
        
        # Add game statistics
        stats_info = self._get_statistics_information(game_state)
        vision_vector.extend(stats_info)
        
        return np.array(vision_vector, dtype=np.float32)
    
    def _cast_vision_ray(self, game_state: GameState, start_x: int, start_y: int,
                        dx: int, dy: int, max_distance: int) -> List[float]:
        """Cast a vision ray in a specific direction"""
        
        wall_distance = max_distance
        body_distance = max_distance
        food_distance = max_distance
        
        for distance in range(1, max_distance + 1):
            x = start_x + dx * distance
            y = start_y + dy * distance
            
            # Check wall collision
            if x < 0 or x >= game_state.width or y < 0 or y >= game_state.height:
                wall_distance = distance
                break
            
            # Check body collision
            if (x, y) in game_state.snake[1:]:  # Exclude head
                body_distance = min(body_distance, distance)
            
            # Check food
            if (x, y) == game_state.food:
                food_distance = min(food_distance, distance)
        
        # Normalize distances
        wall_dist_norm = wall_distance / max_distance
        body_dist_norm = body_distance / max_distance
        food_dist_norm = food_distance / max_distance
        
        return [wall_dist_norm, body_dist_norm, food_dist_norm]
    
    def _get_body_information(self, game_state: GameState) -> List[float]:
        """Get information about snake body"""
        
        if not game_state.snake:
            return [0.0, 0.0, 0.0]
        
        # Snake length (normalized)
        length_norm = min(1.0, len(game_state.snake) / (game_state.width * game_state.height * 0.5))
        
        # Body density around head
        head_x, head_y = game_state.snake[0]
        body_nearby = 0
        
        for radius in range(1, 4):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) == radius:
                        x, y = head_x + dx, head_y + dy
                        if (x, y) in game_state.snake[1:]:
                            body_nearby += 1 / radius  # Closer body parts weighted more
        
        body_density = min(1.0, body_nearby / 12)  # Normalize
        
        # Tail direction relative to head
        if len(game_state.snake) > 1:
            tail_x, tail_y = game_state.snake[-1]
            tail_dx = (tail_x - head_x) / game_state.width
            tail_dy = (tail_y - head_y) / game_state.height
        else:
            tail_dx = tail_dy = 0.0
        
        return [length_norm, body_density, tail_dx, tail_dy]
    
    def _get_food_information(self, game_state: GameState) -> List[float]:
        """Get information about food"""
        
        if not game_state.snake:
            return [0.0, 0.0, 0.0]
        
        head_x, head_y = game_state.snake[0]
        food_x, food_y = game_state.food
        
        # Food position relative to head (normalized)
        food_dx = (food_x - head_x) / game_state.width
        food_dy = (food_y - head_y) / game_state.height
        
        # Food distance (normalized)
        distance = np.sqrt((food_x - head_x)**2 + (food_y - head_y)**2)
        max_distance = np.sqrt(game_state.width**2 + game_state.height**2)
        food_distance = distance / max_distance
        
        return [food_dx, food_dy, food_distance]
    
    def _get_statistics_information(self, game_state: GameState) -> List[float]:
        """Get game statistics information"""
        
        # Score (normalized)
        max_possible_score = game_state.width * game_state.height - 1
        score_norm = game_state.score / max_possible_score
        
        # Steps (normalized)
        max_steps = 1000  # Reasonable game length
        steps_norm = min(1.0, game_state.steps / max_steps)
        
        # Board coverage
        coverage = len(game_state.snake) / (game_state.width * game_state.height)
        
        return [score_norm, steps_norm, coverage]
    
    def _resize_state(self, state: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize state using interpolation"""
        
        if len(state.shape) == 3:
            # Multi-channel state
            resized_channels = []
            for channel in range(state.shape[2]):
                channel_data = state[:, :, channel]
                resized_channel = cv2.resize(
                    channel_data, target_size, interpolation=cv2.INTER_NEAREST
                )
                resized_channels.append(resized_channel)
            
            resized_state = np.stack(resized_channels, axis=2)
        else:
            # Single channel state
            resized_state = cv2.resize(
                state, target_size, interpolation=cv2.INTER_NEAREST
            )
            resized_state = np.expand_dims(resized_state, axis=2)
        
        return resized_state
    
    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        """Apply data augmentation to state"""
        
        augmented_state = state.copy()
        
        # Random rotation (90-degree increments)
        if np.random.random() < 0.5:
            num_rotations = np.random.randint(1, 4)
            for _ in range(num_rotations):
                augmented_state = np.rot90(augmented_state)
        
        # Random flip
        if np.random.random() < 0.5:
            if np.random.random() < 0.5:
                augmented_state = np.fliplr(augmented_state)
            else:
                augmented_state = np.flipud(augmented_state)
        
        # Small noise injection
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, augmented_state.shape)
            augmented_state = np.clip(augmented_state + noise, 0, 1)
        
        return augmented_state
    
    def _add_frame_stacking(self, state: np.ndarray) -> np.ndarray:
        """Add frame stacking for temporal information"""
        
        # Add current frame to buffer
        self.frame_buffer.append(state)
        
        # If buffer not full, pad with zeros
        while len(self.frame_buffer) < self.frame_stack_size:
            self.frame_buffer.appendleft(np.zeros_like(state))
        
        # Stack frames along channel dimension
        stacked_state = np.concatenate(list(self.frame_buffer), axis=2)
        
        return stacked_state
    
    def _update_normalization_params(self, state: np.ndarray):
        """Update running normalization parameters"""
        
        if self.state_mean is None:
            self.state_mean = state.copy()
            self.state_std = np.ones_like(state)
        else:
            # Running average
            alpha = 0.01
            self.state_mean = (1 - alpha) * self.state_mean + alpha * state
            
            # Running standard deviation
            diff = state - self.state_mean
            self.state_std = (1 - alpha) * self.state_std + alpha * np.abs(diff)
    
    def create_attention_mask(self, game_state: GameState) -> np.ndarray:
        """Create attention mask for important regions"""
        
        mask = np.zeros((game_state.height, game_state.width), dtype=np.float32)
        
        if not game_state.snake:
            return mask
        
        head_x, head_y = game_state.snake[0]
        food_x, food_y = game_state.food
        
        # High attention around snake head
        self._add_gaussian_attention(mask, head_x, head_y, sigma=2.0, intensity=1.0)
        
        # High attention around food
        self._add_gaussian_attention(mask, food_x, food_y, sigma=1.5, intensity=0.8)
        
        # Medium attention around snake body
        for segment_x, segment_y in game_state.snake[1:]:
            self._add_gaussian_attention(mask, segment_x, segment_y, sigma=1.0, intensity=0.3)
        
        # Normalize mask
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return mask
    
    def _add_gaussian_attention(self, mask: np.ndarray, center_x: int, center_y: int,
                               sigma: float, intensity: float):
        """Add Gaussian attention around a point"""
        
        height, width = mask.shape
        
        for y in range(height):
            for x in range(width):
                distance_sq = (x - center_x)**2 + (y - center_y)**2
                attention = intensity * np.exp(-distance_sq / (2 * sigma**2))
                mask[y, x] = max(mask[y, x], attention)
    
    def extract_spatial_features(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spatial features from state"""
        
        features = {}
        
        if len(state.shape) < 3:
            return features
        
        # Extract different types of spatial information
        
        # Snake body channel
        if state.shape[2] >= 1:
            body_channel = state[:, :, 0]
            features['body_density'] = self._calculate_density_map(body_channel)
            features['body_connectivity'] = self._calculate_connectivity(body_channel)
        
        # Snake head channel
        if state.shape[2] >= 2:
            head_channel = state[:, :, 1]
            features['head_position'] = self._find_peak_position(head_channel)
        
        # Food channel
        if state.shape[2] >= 3:
            food_channel = state[:, :, 2]
            features['food_position'] = self._find_peak_position(food_channel)
        
        return features
    
    def _calculate_density_map(self, channel: np.ndarray) -> np.ndarray:
        """Calculate local density map"""
        
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        # Apply convolution to get local density
        density_map = cv2.filter2D(channel, -1, kernel)
        
        return density_map
    
    def _calculate_connectivity(self, channel: np.ndarray) -> float:
        """Calculate connectivity of snake body"""
        
        # Find connected components
        binary_channel = (channel > 0.5).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_channel)
        
        # Return connectivity score (1.0 for fully connected)
        if num_labels <= 2:  # Background + one component
            return 1.0
        else:
            return 1.0 / (num_labels - 1)
    
    def _find_peak_position(self, channel: np.ndarray) -> Tuple[int, int]:
        """Find position of maximum value in channel"""
        
        flat_idx = np.argmax(channel)
        y, x = np.unravel_index(flat_idx, channel.shape)
        
        return int(x), int(y)
    
    def get_preprocessing_statistics(self) -> Dict:
        """Get preprocessing statistics"""
        
        stats = {
            'frame_stack_size': self.frame_stack_size,
            'spatial_resolution': self.spatial_resolution,
            'vision_distance': self.vision_distance,
            'augmentation_enabled': self.augment_states,
            'normalization_enabled': self.update_normalization
        }
        
        if self.state_mean is not None:
            stats['state_mean'] = self.state_mean.tolist()
            stats['state_std'] = self.state_std.tolist()
        
        return stats
    
    def reset_preprocessing(self):
        """Reset preprocessing state"""
        
        self.frame_buffer.clear()
        self.state_mean = None
        self.state_std = None
        
        self.logger.info("State preprocessor reset")
    
    def create_state_embedding(self, game_state: GameState) -> np.ndarray:
        """Create a rich state embedding combining multiple representations"""
        
        # Get different state representations
        compact_state = self.preprocess_compact_state(game_state.get_compact_state())
        vision_state = self.preprocess_vision_state(game_state)
        
        # Spatial features
        cnn_state = self.preprocess_cnn_state(game_state.get_state_array())
        spatial_features = self.extract_spatial_features(
            np.transpose(cnn_state, (1, 2, 0))  # Convert back to HWC
        )
        
        # Combine embeddings
        embedding_parts = [compact_state, vision_state]
        
        # Add spatial feature summaries
        for feature_name, feature_data in spatial_features.items():
            if isinstance(feature_data, np.ndarray):
                if feature_data.ndim == 2:
                    # Summarize 2D features
                    feature_summary = [
                        feature_data.mean(),
                        feature_data.std(),
                        feature_data.max(),
                        feature_data.min()
                    ]
                    embedding_parts.append(feature_summary)
                elif feature_data.ndim == 0:
                    # Scalar features
                    embedding_parts.append([float(feature_data)])
            elif isinstance(feature_data, tuple):
                # Position features
                embedding_parts.append(list(feature_data))
        
        # Flatten and concatenate
        embedding = np.concatenate([
            np.array(part).flatten() for part in embedding_parts
        ])
        
        return embedding.astype(np.float32)

class BatchStatePreprocessor:
    """Preprocessor for batch processing of states"""
    
    def __init__(self, base_preprocessor: StatePreprocessor):
        self.base_preprocessor = base_preprocessor
        
    def preprocess_batch(self, states: List[np.ndarray], 
                        state_type: str = 'compact') -> torch.Tensor:
        """Preprocess a batch of states"""
        
        processed_states = []
        
        for state in states:
            if state_type == 'compact':
                processed_state = self.base_preprocessor.preprocess_compact_state(state)
            elif state_type == 'cnn':
                processed_state = self.base_preprocessor.preprocess_cnn_state(state)
            elif state_type == 'vision':
                # Assuming state is a GameState object for vision processing
                processed_state = self.base_preprocessor.preprocess_vision_state(state)
            else:
                raise ValueError(f"Unknown state type: {state_type}")
            
            processed_states.append(processed_state)
        
        # Stack into batch tensor
        batch_tensor = torch.FloatTensor(np.stack(processed_states, axis=0))
        
        return batch_tensor
    
    def create_data_loader(self, states: List[np.ndarray], 
                          batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader for states"""
        
        from torch.utils.data import TensorDataset, DataLoader
        
        # Preprocess all states
        processed_states = self.preprocess_batch(states)
        
        # Create dataset and loader
        dataset = TensorDataset(processed_states)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader
