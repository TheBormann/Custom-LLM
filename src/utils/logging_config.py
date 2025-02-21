import logging
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """Configure logging for the entire application.
    
    Args:
        log_level: The minimum logging level to track
        log_to_file: Whether to save logs to a file
    """
    # Create logs directory if it doesn't exist
    if log_to_file:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific levels for different components
    logging.getLogger('torch').setLevel(logging.WARNING)  # Reduce PyTorch logging
    
    # Log initial configuration
    logging.info(f'Logging configured with level: {logging.getLevelName(log_level)}')
    if log_to_file:
        logging.info(f'Logs will be saved to: {log_file}')