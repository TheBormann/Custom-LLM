import torch
import torch.nn as nn
import math
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CustomAttention(nn.Module):
    """Custom implementation of scaled dot-product attention with additional features.
    
    This class implements an enhanced attention mechanism that extends the basic transformer
    attention with additional flexibility and features:
    Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    
    Args:
        d_model (int): The input/output dimension
        n_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.1)
        attention_scale (float): Custom scaling factor for attention scores (default: None)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, attention_scale: Optional[float] = None):
        super().__init__()
        
        assert d_model % n_heads == 0, 'Model dimension must be divisible by number of heads'
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.attention_scale = attention_scale or 1.0 / math.sqrt(self.d_k) # defined by the paper
        
        # Linear projections with Xavier initialization
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation: Searches for a specific "label"
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation: is the "label" for and embedding
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation: contains information
        self.W_o = nn.Linear(d_model, d_model)  # Output projection: converts value to embedding space
        
        # Initialize weights with scaled Xavier initialization
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            # Scale the initialization by sqrt(2.0 / (d_model + d_model)) for better gradient flow
            gain = math.sqrt(2.0 / (d_model + d_model))
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            # Initialize biases to small values instead of zeros
            nn.init.uniform_(layer.bias, -0.1 / math.sqrt(d_model), 0.1 / math.sqrt(d_model))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute scaled dot-product attention with enhanced features.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor of shape [batch_size, seq_len_q, seq_len_k]
            return_attention: Whether to return attention weights (default: False)
            
        Returns:
            output: Attention output of shape [batch_size, seq_len_q, d_model]
            attention_weights: Optional attention weights if return_attention is True
        """
        batch_size = query.size(0)
        logger.debug(f"Processing attention with shapes - Query: {query.shape}, Key: {key.shape}, Value: {value.shape}")
        
        # Linear projections and reshape for multi-head attention
        # We leave seq_len flexible so we can allow variable context windows
        # We transpose to shape [batch_size, n_heads, seq_len, d_k] to allow for parallelization
        # this is because in memory these columns are close to each other and allow for an instant look up of an unique index of batch + head without mixup
        
        # Query: "What information should I look for?"
        Q = self.W_q(query)
        logger.info(f"Query projection stats - Mean: {Q.mean():.4f}, Std: {Q.std():.4f}, Max: {Q.max():.4f}, Min: {Q.min():.4f}")
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Key: "What information do I contain?"
        K = self.W_k(key)
        logger.info(f"Key projection stats - Mean: {K.mean():.4f}, Std: {K.std():.4f}, Max: {K.max():.4f}, Min: {K.min():.4f}")
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Value: "What information should I pass on?"
        V = self.W_v(value)
        logger.info(f"Value projection stats - Mean: {V.mean():.4f}, Std: {V.std():.4f}, Max: {V.max():.4f}, Min: {V.min():.4f}")
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        logger.debug(f"Projected shapes - Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
        
        # dot-product -> attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.attention_scale
        logger.info(f"Raw attention scores stats - Mean: {scores.mean():.4f}, Std: {scores.std():.4f}, Max: {scores.max():.4f}, Min: {scores.min():.4f}")
        logger.info(f"Attention scores shape: {scores.shape}, scale: {self.attention_scale}")
        
        # Apply mask to prevent attention to certain positions and handling of padding
        if mask is not None:
            # Expand mask for multi-head attention [batch_size, 1, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            logger.info(f"Applied mask shape: {mask.shape}")
            
            # Instead of applying masked_fill again, just add the mask (since it already has -inf)
            scores = scores + mask
        
        # Compute attention weights and apply dropout
        attention_weights = torch.softmax(scores, dim=-1)
        logger.info(f"Attention weights stats - Mean: {attention_weights.mean():.4f}, Std: {attention_weights.std():.4f}, Max: {attention_weights.max():.4f}, Sparsity: {(attention_weights < 0.01).float().mean():.4f}")
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        logger.debug(f"Context shape after attention: {context.shape}")
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        logger.debug(f"Final output shape: {output.shape}")
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)