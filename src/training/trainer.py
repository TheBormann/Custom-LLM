"""Training pipeline for the Custom LLM model.

This module implements the training loop, optimization, and validation
for the transformer model.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

class Trainer:
    """Handles model training and validation."""
    
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 warmup_steps: int = 4000,
                 max_grad_norm: float = 1.0,
                 use_wandb: bool = False):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        
        # Initialize optimizer
        self.optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss() # well-suited for sequence-to-sequence tasks
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              epochs: int,
              save_path: Optional[str] = None,
              log_interval: int = 100) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of training epochs
            save_path: Optional path to save model checkpoints
            log_interval: Steps between logging
            
        Returns:
            Dictionary containing training history
        """
        if self.use_wandb:
            wandb.init(project="custom-llm", config={
                "learning_rate": self.learning_rate,
                "warmup_steps": self.warmup_steps,
                "max_grad_norm": self.max_grad_norm,
                "epochs": epochs
            })
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': []
        }
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training loop
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs, _ = self.model(input_ids, attention_mask)
                
                # Calculate loss (shift predictions and targets)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1))
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Log metrics
                if global_step % log_interval == 0:
                    avg_loss = train_loss / train_steps
                    perplexity = torch.exp(torch.tensor(avg_loss))
                    
                    if self.use_wandb:
                        wandb.log({
                            'train_loss': avg_loss,
                            'train_perplexity': perplexity,
                            'learning_rate': self.scheduler.get_last_lr()[0]
                        }, step=global_step)
            
            # Calculate average training metrics
            avg_train_loss = train_loss / train_steps
            train_perplexity = torch.exp(torch.tensor(avg_train_loss))
            
            # Validation loop
            val_loss, val_perplexity = self.evaluate(val_dataloader)
            
            # Store metrics
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['train_perplexity'].append(train_perplexity.item())
            history['val_perplexity'].append(val_perplexity.item())
            
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Average Train Loss: {avg_train_loss:.4f}')
            print(f'Train Perplexity: {train_perplexity:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Perplexity: {val_perplexity:.4f}')
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
        
        if self.use_wandb:
            wandb.finish()
        
        return history
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[float, torch.Tensor]:
        """Evaluate the model on validation data.
        
        Args:
            dataloader: DataLoader for validation data
            
        Returns:
            Tuple of (average loss, perplexity)
        """
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs, _ = self.model(input_ids, attention_mask)
            
            # Calculate loss
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1))
            
            total_loss += loss.item()
            total_steps += 1
        
        avg_loss = total_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return avg_loss, perplexity