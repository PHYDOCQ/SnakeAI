"""
CNN+LSTM+Attention hybrid architecture for Snake AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for CNN features"""
    
    def __init__(self, in_channels: int):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention

class CNNFeatureExtractor(nn.Module):
    """CNN for extracting spatial features from game state"""
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 256):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Third conv block with attention
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Spatial attention
        self.attention = SpatialAttention(128)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection
        self.feature_proj = nn.Linear(128, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        features = self.conv_layers(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Project to feature dimension
        features = self.feature_proj(pooled)
        
        return features

class TemporalLSTM(nn.Module):
    """LSTM for temporal sequence modeling"""
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2):
        super(TemporalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (batch, sequence, features)
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        return lstm_out, hidden

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Output projection
        output = self.out_proj(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x)
        
        return output

class CNNLSTMAttentionDQN(nn.Module):
    """Hybrid CNN+LSTM+Attention model for Snake AI"""
    
    def __init__(self, input_shape: Tuple[int, int, int], 
                 sequence_length: int = 8,
                 cnn_feature_dim: int = 256,
                 lstm_hidden_size: int = 256,
                 attention_heads: int = 8,
                 output_size: int = 4):
        super(CNNLSTMAttentionDQN, self).__init__()
        
        self.sequence_length = sequence_length
        height, width, channels = input_shape
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(channels, cnn_feature_dim)
        
        # LSTM for temporal modeling
        self.lstm = TemporalLSTM(cnn_feature_dim, lstm_hidden_size)
        
        # Multi-head attention
        self.attention = MultiHeadSelfAttention(lstm_hidden_size, attention_heads)
        
        # Decision head
        self.decision_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden_size // 2, output_size)
        )
        
        # Value head for dueling architecture
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden_size // 2, 1)
        )
        
        # State buffer for sequence modeling
        self.register_buffer('state_buffer', torch.zeros(1, sequence_length, cnn_feature_dim))
        self.hidden_state = None
    
    def forward(self, x: torch.Tensor, reset_hidden: bool = False) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Reset hidden state if requested
        if reset_hidden or self.hidden_state is None:
            self.hidden_state = None
            self.state_buffer.zero_()
        
        # Extract CNN features
        cnn_features = self.cnn(x)  # (batch, feature_dim)
        
        # Update state buffer
        if batch_size == 1:  # Single sample - update buffer
            self.state_buffer = torch.roll(self.state_buffer, -1, dims=1)
            self.state_buffer[0, -1] = cnn_features[0]
            sequence_input = self.state_buffer
        else:  # Batch processing - create sequences
            sequence_input = cnn_features.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # LSTM processing
        lstm_out, self.hidden_state = self.lstm(sequence_input, self.hidden_state)
        
        # Attention over LSTM outputs
        attended_features = self.attention(lstm_out)
        
        # Use last timestep for decision making
        final_features = attended_features[:, -1]
        
        # Dueling architecture
        advantage = self.decision_head(final_features)
        value = self.value_head(final_features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization"""
        with torch.no_grad():
            cnn_features = self.cnn(x)
            
            if x.size(0) == 1:
                self.state_buffer = torch.roll(self.state_buffer, -1, dims=1)
                self.state_buffer[0, -1] = cnn_features[0]
                sequence_input = self.state_buffer
            else:
                sequence_input = cnn_features.unsqueeze(1).repeat(1, self.sequence_length, 1)
            
            lstm_out, _ = self.lstm(sequence_input, self.hidden_state)
            
            # Get attention weights from the attention module
            # This would require modifying the attention module to return weights
            return torch.ones(sequence_input.size(0), self.sequence_length)  # Placeholder
    
    def reset_sequence(self):
        """Reset the sequence buffer and hidden state"""
        self.state_buffer.zero_()
        self.hidden_state = None

class VisionTransformerDQN(nn.Module):
    """Vision Transformer-based DQN"""
    
    def __init__(self, input_shape: Tuple[int, int, int],
                 patch_size: int = 4,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 output_size: int = 4):
        super(VisionTransformerDQN, self).__init__()
        
        height, width, channels = input_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (height // patch_size) * (width // patch_size)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Create patches
        patches = self.patch_embed(x)  # (batch, embed_dim, H/p, W/p)
        patches = patches.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use class token for classification
        cls_output = x[:, 0]
        
        # Final prediction
        output = self.head(cls_output)
        
        return output
