from .attention import CustomAttention
from .position_encoding import PositionalEncoding
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class CustomTransformer(nn.Module):
    """Custom transformer model with reasoning capabilities.
    
    This implementation follows a simplified transformer architecture with
    additional components for enhanced reasoning capabilities.
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_seq_length: int = 1024):
        super().__init__()
        
        # Store model dimensions and log initialization
        self.d_model = d_model
        logger.info(f"Initializing CustomTransformer with d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Ensure d_model is passed correctly to transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        logger.info(f"Created {n_layers} transformer layers with d_ff={d_ff}, dropout={dropout}")
        
        self.final_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None, return_attention=False):
        """Process input through the transformer model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length]
            mask: Optional attention mask
            return_attention: Whether to return attention weights (default: False)
            
        Returns:
            output: Output logits of shape [batch_size, seq_length, vocab_size]
            attention_weights: Optional list of attention weights if return_attention is True
        """
        logger.debug(f"Input shape: {x.shape}, mask shape: {mask.shape if mask is not None else None}")
        # Embedding and positional encoding
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.embedding.embedding_dim))
        logger.debug(f"Embedding output shape: {x.shape}")
        
        x = self.pos_encoding(x)
        logger.debug(f"Positional encoding output shape: {x.shape}")
        
        # Process through transformer layers
        attention_weights_list = []
        layer_norms = []
        for i, layer in enumerate(self.transformer_layers):
            x, attention_weights = layer(x, mask, return_attention)
            logger.debug(f"Layer {i} output shape: {x.shape}")
            # Log layer statistics
            with torch.no_grad():
                layer_mean = x.mean().item()
                layer_std = x.std().item()
                layer_norms.append(torch.norm(x).item())
                logger.info(f"Layer {i} stats - Mean: {layer_mean:.4f}, Std: {layer_std:.4f}, Norm: {layer_norms[-1]:.4f}")
            
            if return_attention:
                attention_weights_list.append(attention_weights)
                if attention_weights is not None:
                    logger.info(f"Layer {i} attention pattern - Mean: {attention_weights.mean().item():.4f}, Sparsity: {(attention_weights < 0.01).float().mean().item():.4f}")
                    logger.debug(f"Layer {i} attention weights shape: {attention_weights.shape}")
        
        # Final linear projection to vocabulary size
        output = self.final_layer(x)
        logger.debug(f"Final output shape: {output.shape}")
        
        if return_attention:
            return output, attention_weights_list
        return output, None

class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward network.
    
    Implements the standard transformer layer with multi-head self-attention
    followed by a position-wise feed-forward network. Includes residual
    connections and layer normalization.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = CustomAttention(d_model, n_heads, dropout=dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),     # Expand dimension (e.g., 512 -> 2048)
            nn.ReLU(),                    # Non-linearity
            nn.Dropout(dropout),          # Prevent overfitting
            nn.Linear(d_ff, d_model)      # Project back to model dimension (2048 -> 512)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask, return_attention=False):
        """Process input through self-attention and feed-forward layers.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Processed tensor of same shape as input
            Optional attention weights if return_attention is True
        """
        
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attn(x, x, x, mask=mask, return_attention=return_attention)
        
        x = x + self.dropout(attn_output) 
        x = self.norm1(x) 
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        
        x = x + self.dropout(ff_output)
        x = self.norm2(x) 
        
        if return_attention:
            return x, attention_weights
        return x, None