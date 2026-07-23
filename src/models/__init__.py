from .simple_cnn import SimpleCNN
from .unet import UNet
from .siamese_temporal_attention_unet import SiameseTemporalAttentionUNet


MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "unet": UNet,
    "siamese_temporal_attention_unet": SiameseTemporalAttentionUNet,
}


def build_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name]()
