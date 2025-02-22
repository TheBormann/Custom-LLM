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
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        
        if save_path:
            try:
                last_epoch, best_val_loss = self.load_checkpoint(save_path)
                logger.info(f"Resuming training from epoch {last_epoch+1} with best validation loss {best_val_loss:.4f}")
            except FileNotFoundError:
                logger.info("No checkpoint found, starting training from scratch.")
                last_epoch = 0
                best_val_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Training samples: {len(train_dataloader.dataset)}")
        logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
        
        if self.use_wandb:
            # Initialize W&B with comprehensive config
            model_config = {
                name: getattr(self.model, name, None)
                for name in ['d_model', 'n_heads', 'n_layers', 'd_ff', 'vocab_size']
                if hasattr(self.model, name)
            }
            
            wandb.init(
                project="custom-llm",
                config={
                    "model_architecture": type(self.model).__name__,
                    "model_parameters": sum(p.numel() for p in self.model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    "learning_rate": self.learning_rate,
                    "warmup_steps": self.warmup_steps,
                    "max_grad_norm": self.max_grad_norm,
                    "epochs": epochs,
                    "optimizer": type(self.optimizer).__name__,
                    "scheduler": "WarmupLR",
                    "loss_function": type(self.criterion).__name__,
                    "device": str(self.device),
                    **model_config
                }
            )
            
            # Log model architecture as a summary
            wandb.watch(self.model, log="all", log_freq=log_interval)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': []
        }
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            
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
                
                # Check for NaN in model outputs
                if torch.isnan(outputs).any():
                    logger.warn("Warning: NaN detected in model outputs")
                    continue
                
                # Calculate loss (shift predictions and targets)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Add numerical stability
                shift_logits = torch.clamp(shift_logits, min=-100, max=100)  # Prevent extreme values
                
                loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1))
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    logger.warn("Warning: NaN detected in loss calculation")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN in gradients
                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    logger.warn("Warning: NaN detected in gradients")
                    continue
                
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
                        # Log comprehensive training metrics
                        # Compute attention patterns
                        _, attention_weights = self.model(input_ids, attention_mask)
                        
                        # Basic training metrics
                        metrics = {
                            'train/loss': avg_loss,
                            'train/perplexity': perplexity,
                            'train/learning_rate': self.scheduler.get_last_lr()[0],
                            'train/grad_norm': torch.nn.utils.clip_grad_norm_(self.model.parameters(), -1).item(),
                        }
                        
                        # Track attention patterns
                        if attention_weights is not None:
                            for layer_idx, attn in enumerate(attention_weights):
                                metrics[f'train/attention/layer_{layer_idx}/pattern'] = wandb.Image(
                                    attn[0].mean(0).cpu().detach().numpy(),
                                    caption=f'Layer {layer_idx} Attention Pattern'
                                )
                        
                        # Track activation statistics
                        for name, module in self.model.named_modules():
                            if isinstance(module, nn.Linear):
                                if hasattr(module, 'weight'):
                                    metrics.update({
                                        f'train/activations/{name}/weight_mean': module.weight.data.mean().item(),
                                        f'train/activations/{name}/weight_std': module.weight.data.std().item(),
                                        f'train/activations/{name}/weight_norm': module.weight.data.norm().item()
                                    })
                        
                        # Track gradient flow
                        grad_flow = {}
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                grad_flow[f'train/grad_flow/{name}'] = grad_norm
                        metrics.update(grad_flow)
                        
                        # Track token prediction accuracy
                        with torch.no_grad():
                            predictions = shift_logits.argmax(dim=-1)
                            correct_predictions = (predictions == shift_labels).float().mean().item()
                            metrics['train/token_accuracy'] = correct_predictions
                        
                        # Log parameter statistics
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                metrics.update({
                                    f'train/params/{name}/mean': param.data.mean().item(),
                                    f'train/params/{name}/std': param.data.std().item(),
                                    f'train/grads/{name}/mean': param.grad.mean().item() if param.grad is not None else 0,
                                    f'train/grads/{name}/std': param.grad.std().item() if param.grad is not None else 0
                                })
                        
                        wandb.log(metrics, step=global_step)
            
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
            
            # Replace existing print statements with logger
            logger.info(f"Epoch {epoch + 1}/{epochs} Results:")
            logger.info(f"Average Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Train Perplexity: {train_perplexity:.4f}")
            logger.info(f"Validation Loss: {val_loss:.4f}")
            logger.info(f"Validation Perplexity: {val_perplexity:.4f}")
            
            if save_path and val_loss < best_val_loss:
                logger.info(f"New best model saved to {save_path}")
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
            
        Raises:
            ValueError: If the dataloader is empty
        """
        # Validate dataloader
        if len(dataloader) == 0:
            raise ValueError("Validation dataloader is empty. Please ensure your validation dataset contains data.")
            
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        for batch in dataloader:
            # Validate batch data
            if not batch or 'input_ids' not in batch or 'attention_mask' not in batch:
                continue
                
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Skip empty batches
            if input_ids.size(0) == 0:
                continue
            
            outputs, _ = self.model(input_ids, attention_mask)
            
            # Calculate loss
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1))
            
            total_loss += loss.item()
            total_steps += 1
        
        # Handle case where no valid batches were processed
        if total_steps == 0:
            raise ValueError("No valid batches found in validation dataloader. Please check your data preprocessing.")
        
        avg_loss = total_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return avg_loss, perplexity
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model, optimizer, and scheduler from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']