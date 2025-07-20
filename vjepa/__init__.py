# V-JEPA2-AC components for 1X challenge

from .config import VJEPAEncoderConfig, VJEPAPredictorConfig
from .encoder import VJEPAEncoder
from .predictor import VJEPAPredictor


def create_vjepa_predictor(config_path: str):
    """Create V-JEPA predictor from config file"""
    config = VJEPAPredictorConfig.from_pretrained(config_path)
    return VJEPAPredictor(config)
