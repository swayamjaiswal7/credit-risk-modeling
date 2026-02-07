import yaml
import logging
from pathlib import Path
import joblib

def setup_logging(log_file=None, log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML"""
    if not Path(config_path).exists():
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def ensure_dir(directory):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def save_model(model, filepath, metadata=None):
    """Save model with metadata"""
    ensure_dir(Path(filepath).parent)
    
    if metadata:
        save_obj = {'model': model, 'metadata': metadata}
    else:
        save_obj = model
    
    joblib.dump(save_obj, filepath)
