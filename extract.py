"""Extract mono 16 kHz speech in bounded blocks directly from a WAV stream."""
import argparse
from pathlib import Path

import soundfile as sf
import torch

from streaming import Stream
from tse_model import ROOT, load_model, enrollment


@torch.inference_mode()
def extract(args):
    torch.set_num_threads(args.threads)
    if args.output.resolve() in {args.mixture.resolve(), args.enroll.resolve()}:
        raise ValueError("output must differ from both inputs")
    if args.chunk < 1:
        raise ValueError("chunk must be positive")
    ref, sr = sf.read(args.enroll, dtype='float32')
    if sr != 16000 or ref.ndim != 1:
        raise ValueError("enrollment must be mono 16 kHz")
    model = load_model(args.checkpoint)
    embedding = enrollment(model, torch.from_numpy(ref))
    if args.mask_period != 1 and args.backend != "ort-int8":
        raise ValueError("--mask-period is only supported by ort-int8")
    if args.backend == "openvino":
        from cpu_engine import CpuEngine
        stream = CpuEngine(model, args.threads).stream(embedding)
    elif args.backend == "ort-int8":
        from ort_engine import OrtEngine, verify_graph
        verify_graph(args.graph, args.checkpoint)
        stream = OrtEngine(model, args.graph, args.threads, args.mask_period).stream(embedding)
    else:
        stream = Stream(model, embedding)
    with sf.SoundFile(args.mixture) as src:
        if src.samplerate != 16000 or src.channels != 1 or len(src) <= 256:
            raise ValueError("expected mono 16 kHz mixture longer than 256 samples")
        with sf.SoundFile(args.output, mode='w', samplerate=16000, channels=1, subtype='FLOAT') as dst:
            for block in src.blocks(blocksize=args.chunk, dtype='float32'):
                dst.write(stream.push(torch.from_numpy(block)).numpy())
            dst.write(stream.flush().numpy())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('mixture','enroll','output'):
        p.add_argument(key,type=Path)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'assets/spk_emb_causal_100')
    p.add_argument('--chunk',type=int,default=128,help="input block size in samples")
    p.add_argument('--threads',type=int,default=2)
    p.add_argument("--backend",choices=["torch","openvino","ort-int8"],default="torch")
    p.add_argument("--graph",type=Path,default=ROOT/"assets/onnx/one_hop_int8.onnx")
    p.add_argument("--mask-period",type=int,default=1,help="INT8 mask update interval in hops; >1 changes the model")
    extract(p.parse_args())
