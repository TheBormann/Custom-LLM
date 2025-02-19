import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input using sinusoidal functions.
    
    Creates unique position-dependent patterns that are added to token embeddings.
    Each position gets a distinct encoding while maintaining relative positional
    relationships through sinusoidal patterns of different frequencies:
    - High frequencies (lower dimensions) capture local, short-range relationships
    - Low frequencies (higher dimensions) capture global, long-range relationships
    
    This implementation follows the paper 'Attention Is All You Need',
    using sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        wavelength = 10000.0 # Defined in the paper
        
        pe = torch.zeros(max_seq_length, d_model)
        # Create position indices
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Create frequency that starts high and decreases exponentially 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(wavelength)) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]