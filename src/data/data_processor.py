"""Data processing module for the LLM.

This module handles dataset loading, preprocessing, and tokenization for both
basic language modeling and reasoning-specific tasks.
"""

from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class CustomDataset(Dataset):
    """Custom dataset for transformer training."""
    
    def __init__(self,
                 texts: List[str],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Configure padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(text,
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')
        
        # Get the base attention mask from tokenizer
        attention_mask = encoding['attention_mask'].squeeze()
        seq_length = attention_mask.size(0)
        
        # Create causal mask (for autoregressive attention)
        # Shape will be expanded to [batch_size, n_heads, seq_len, seq_len] during forward pass
        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        attention_mask = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)
        attention_mask = attention_mask.masked_fill(causal_mask, 0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': attention_mask
        }

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024,
                 batch_size: int = 32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
    
    def prepare_data(self,
                     texts: List[str],
                     split_ratio: float = 0.1
                     ) -> tuple[DataLoader, DataLoader]:
        """Prepare train and validation dataloaders.
        
        Args:
            texts: List of input texts
            split_ratio: Ratio of validation split (default: 0.1)
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Calculate split index
        val_size = int(len(processed_texts) * split_ratio)
        train_size = len(processed_texts) - val_size
        
        # Split into train and validation sets
        train_texts = processed_texts[:train_size]
        val_texts = processed_texts[train_size:]
        
        # Create datasets
        train_dataset = CustomDataset(train_texts, self.tokenizer, self.max_length)
        val_dataset = CustomDataset(val_texts, self.tokenizer, self.max_length)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_dataloader, val_dataloader
    
    def preprocess_text(self, text: str) -> str:
        """Apply text preprocessing steps.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = "".join(char for char in text if char.isalnum() or char in " .,!?-'\"")
        
        return text.strip()