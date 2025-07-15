# V-JEPA2-AC components for 1X challenge

from .config import VJEPAEncoderConfig, VJEPAPredictorConfig, VJEPAConfig  # Legacy alias
from .predictor import VJEPAPredictor
from .encoder import VJEPAEncoder, create_vjepa_encoder

def create_vjepa_predictor(config_path: str):
    """Create V-JEPA predictor from config file"""
    config = VJEPAPredictorConfig.from_pretrained(config_path)
    return VJEPAPredictor(config)

