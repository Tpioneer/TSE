"""Inference-only dispatch for the sole encoder in this checkpoint."""
from .ecapa_tdnn import ECAPA_TDNN_GLOB_c512


def get_speaker_model(name):
    if name != "ECAPA_TDNN_GLOB_c512":
        raise ValueError(f"Unsupported speaker model: {name}")
    return ECAPA_TDNN_GLOB_c512
