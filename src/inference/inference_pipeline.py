"""Inference pipeline for the Custom LLM model.

This module handles model loading, text generation, and efficient inference
for the transformer model.
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from src.model.transformer import CustomTransformer

class InferencePipeline:
    """Handles model inference and text generation."""
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_pretrained(cls,
                       model_path: str,
                       tokenizer: PreTrainedTokenizer,
                       device: Optional[str] = None) -> 'InferencePipeline':
        """Load a pretrained model from disk.
        
        Args:
            model_path: Path to the saved model checkpoint
            tokenizer: Pretrained tokenizer
            device: Device to load the model on
            
        Returns:
            InferencePipeline instance
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize model with same config
        model = CustomTransformer(
            vocab_size=len(tokenizer),
            d_model=512,  # These should match training config
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            dropout=0.1
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, tokenizer, device)
    
    @torch.no_grad()
    def generate(self,
                prompt: str,
                max_length: int = 100,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.95,
                num_return_sequences: int = 1) -> List[str]:
        """Generate text based on a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text sequences
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Track generated sequences
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            # Initialize sequence with input
            curr_input_ids = input_ids.clone()
            
            # Generate tokens one at a time
            for _ in range(max_length):
                # Get model predictions
                outputs, _ = self.model(curr_input_ids)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                
                # Apply top-p filtering (nucleus sampling)
                probs = torch.softmax(top_k_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                tokens_to_remove = cumulative_probs > top_p
                tokens_to_remove[:, 1:] = tokens_to_remove[:, :-1].clone()
                tokens_to_remove[:, 0] = 0
                
                # Set removed tokens to negative infinity
                top_k_logits[tokens_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_idx)
                
                # Append to sequence
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
                
                # Check if end of sequence token is generated
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # Decode generated sequence
            generated_text = self.tokenizer.decode(curr_input_ids[0],
                                                 skip_special_tokens=True)
            generated_sequences.append(generated_text)
        
        return generated_sequences
    
    def batch_generate(self,
                      prompts: List[str],
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """Generate text for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for generation
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            List of generated sequences
        """
        all_sequences = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            for prompt in batch_prompts:
                sequences = self.generate(prompt, **kwargs)
                all_sequences.extend(sequences)
        
        return all_sequences