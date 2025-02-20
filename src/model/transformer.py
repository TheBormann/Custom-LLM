from model.attention import CustomAttention
from model.position_encoding import PositionalEncoding
import torch
import torch.nn as nn

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
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        """Process input through the transformer model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length]
            mask: Optional attention mask
            
        Returns:
            Output logits of shape [batch_size, seq_length, vocab_size]
        """
        # Embedding and positional encoding
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.embedding.embedding_dim))
        x = self.pos_encoding(x)
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Final linear projection to vocabulary size
        output = self.final_layer(x)
        
        return output

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
        
    def forward(self, x, mask=None):
        """Process input through self-attention and feed-forward layers.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            mask: Optional attention mask
            
        Returns:
            Processed tensor of same shape as input
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x