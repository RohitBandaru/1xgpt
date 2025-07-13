# V-JEPA2-AC based world model for 1X challenge

from .config import VJEPAConfig
from .model import VJEPAWorldModel

def create_vjepa_model(config_path: str):
    """Create V-JEPA model from config file"""
    config = VJEPAConfig.from_pretrained(config_path)
    return VJEPAWorldModel(config)