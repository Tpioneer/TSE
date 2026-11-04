"""Measure exclusive stage wall time on live 128-sample push calls."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from benchmark import read
from streaming import Stream
from tse_model import ROOT, enrollment, load_model


@torch.inference_mode()
def profile(threads, frames):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = load_model()
    audio = read(ROOT / "examples/mixture.wav")
    if threads < 1 or frames < 1 or 1024+frames*128 > len(audio):
        raise ValueError("Profile frame count must fit the available fixture")
    stream = Stream(model, enrollment(model, read(ROOT / "examples/enroll.wav")))
    groups = defaultdict(list)
    sep = model.sep_model
    groups["analysis_bands"] = [sep.subband_norm]
    groups["mask_heads"] = [sep.band_masker]
    groups["registration_fusion"] = [stream.fuse.fusionLayer]
    for block in sep.separator.separation:
        groups["time_lstm"].append(block.band_rnn.rnn)
        groups["time_norm_projection"].extend([block.band_rnn.norm, block.band_rnn.proj])
        groups["frequency_bilstm"].append(block.band_comm.rnn)
        groups["frequency_norm_projection"].extend([block.band_comm.norm, block.band_comm.proj])
    stream.push(audio[:1024])
    cumulative, starts, hooks = defaultdict(float), {}, []
    def before(module, _):
        starts[module] = time.perf_counter()
    for group, modules in groups.items():
        def after(module, inputs, outputs, name=group):
            cumulative[name] += time.perf_counter() - starts[module]
        for module in modules:
            hooks.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    start = time.perf_counter()
    for i in range(frames):
        offset = 1024 + i*128
        stream.push(audio[offset:offset+128])
    total = time.perf_counter() - start
    for hook in hooks:
        hook.remove()
    cumulative["fft_ola_dispatch_other"] = total - sum(cumulative.values())
    return {"threads":threads, "frames":frames, "total_ms_per_frame":1000*total/frames,
            "stage_ms_per_frame":{k:1000*v/frames for k,v in cumulative.items()}}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--frames",type=int,default=32)
    p.add_argument("--output",type=Path,default=ROOT/"results/cpu_profile.json")
    args=p.parse_args()
    result=profile(args.threads,args.frames)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2))
