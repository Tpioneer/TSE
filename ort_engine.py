"""Single-hop ONNX Runtime CPU inference. Registration and waveform I/O stay in FP32."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from streaming import Stream


class OrtEngine:
    def __init__(self, model, path, threads, mask_period=1):
        if mask_period < 1 or threads < 1:
            raise ValueError("threads and mask_period must be positive")
        self.mask_period = mask_period
        self.model = model
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def stream(self, embedding):
        return OrtStream(self, embedding)


class OrtStream(Stream):
    def __init__(self, engine, embedding):
        super().__init__(engine.model, embedding)
        self.session = engine.session
        self.mask_period = engine.mask_period
        self.phase = 0
        self.mask = None
        gate = self.fuse.post(torch.ones(1,1,self.sep.feature_dim,1), embedding)
        self.inputs = {"gate": gate.detach().reshape(1,-1).contiguous().numpy()}
        self.states = [port.name for port in self.session.get_inputs()[2:]]
        for port in self.session.get_inputs()[2:]:
            self.inputs[port.name] = np.zeros(port.shape, np.float32)

    def _mask(self, spec):
        outputs = []
        for frame in spec[0].T:
            if self.phase == 0:
                self.inputs["spectrum"] = torch.stack((frame.real,frame.imag)).contiguous().numpy()
                mask, *states = self.session.run(None, self.inputs)
                self.inputs.update(zip(self.states, states))
                self.mask = torch.from_numpy(mask[0] + 1j*mask[1])
            outputs.append(frame * self.mask)
            self.phase = (self.phase + 1) % self.mask_period
        return torch.stack(outputs)

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_graph(path, checkpoint):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
    if manifest != {"graph_sha256": digest(path), "checkpoint_sha256": digest(checkpoint/"avg_model.pt")}:
        raise ValueError("Model or ONNX graph differs from export manifest; rerun export_cpu.py")
