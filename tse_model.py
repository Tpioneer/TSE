"""Inference-only loader: original model, original config and strict official state dict."""
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party"))
from wesep.models.tse_bsrnn_spk import TSE_BSRNN_SPK


def load_model(directory=ROOT / "assets/spk_emb_causal_100"):
    directory = Path(directory)
    with (directory / "config.yaml").open(encoding="utf8") as stream:
        config = yaml.safe_load(stream)["model_args"]["tse_model"]
    sep = config["separator"]
    active = {name for name, feature in config["speaker"]["features"].items()
              if feature.get("enabled")}
    if not sep["causal"] or active != {"spkemb"} or (sep["sr"], sep["win"], sep["stride"]) != (16000, 512, 128):
        raise ValueError("Expected the 16 kHz spk_emb_causal_100 configuration")
    # The published state dict already includes the enrollment network.
    config["speaker"]["speaker_model"]["speaker_encoder"]["pretrained"] = None
    model = TSE_BSRNN_SPK(config)
    state = torch.load(directory / "avg_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["models"][0], strict=True)
    return model.eval()


def enrollment(model, reference):
    torch.manual_seed(42)
    return model.spk_ft.spkemb.compute(reference.reshape(1, -1))[:, None, :, None]
