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
    stream = Stream(model, enrollment(model, torch.from_numpy(ref)))
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
    extract(p.parse_args())
