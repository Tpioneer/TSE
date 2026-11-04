"""Reproduce official forward, streaming equality, suffix tests and CPU latency."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from streaming import Stream
from tse_model import ROOT, enrollment, load_model

SR = 16000


def read(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim != 1 or sr != SR:
        raise ValueError(f"{path}: expected mono 16 kHz")
    return torch.from_numpy(x)


def sisdr(y, target):
    y, target = y.double(), target[:len(y)].double()
    y, target = y-y.mean(), target-target.mean()
    projection = target * (y @ target) / (target @ target)
    return float(10 * torch.log10(projection.square().sum() / (y-projection).square().sum()))


def reference(model, x, enroll):
    torch.manual_seed(42)
    return model(x[None], enroll[None]).flatten()


def stream_run(model, emb, x, size):
    start = time.perf_counter()
    stream = Stream(model, emb)
    out, service, first_at, ready_ms = [], [], None, 0.0
    # Simulate an arriving audio callback with measured (unpaced) service times.
    for offset in range(0, len(x), size):
        block = x[offset:offset+size]
        before = time.perf_counter()
        output = stream.push(block)
        ms = 1000*(time.perf_counter()-before)
        arrival_ms = 1000*(offset+len(block))/SR
        ready_ms = max(ready_ms, arrival_ms) + ms
        if output.numel() and first_at is None:
            first_at = (arrival_ms, ready_ms)
        out.append(output)
        if len(block)==size and output.numel():
            service.append(ms)
    out.append(stream.flush())
    y = torch.cat(out)
    return y, (time.perf_counter()-start)/(len(x)/SR), service, first_at


def paced_first(model, emb, x, size):
    stream = Stream(model, emb)
    start = time.perf_counter()
    for offset in range(0,len(x),size):
        block=x[offset:offset+size]
        until=start+(offset+len(block))/SR-time.perf_counter()
        if until>0:
            time.sleep(until)
        y=stream.push(block)
        if y.numel():
            return 1000*(time.perf_counter()-start)
    raise RuntimeError("No output before EOS")


def error(y,ref):
    diff=(y-ref).double()
    return dict(length=len(y), max_abs=diff.abs().max().item(),rms=diff.square().mean().sqrt().item(),
                snr_db=float(10*torch.log10(ref.double().square().sum()/diff.square().sum())))


@torch.inference_mode()
def main(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    x, enroll = read(args.mixture), read(args.enroll)
    dest=args.output
    dest.mkdir(parents=True,exist_ok=True)
    model=load_model(args.checkpoint)
    print("Official loaded",flush=True)
    ref=reference(model,x,enroll)
    sf.write(dest/'whole.wav',ref.numpy(),SR,subtype='FLOAT')
    before=time.perf_counter()
    emb=enrollment(model,enroll)
    enroll_ms=(time.perf_counter()-before)*1000
    with (args.checkpoint/'avg_model.pt').open('rb') as weights:
        checkpoint_hash = hashlib.file_digest(weights, 'sha256').hexdigest()
    results=dict(torch=torch.__version__,python=platform.python_version(),cpu=platform.processor(),
                 threads=args.threads, sample_rate=SR,input_samples=len(x),output_samples=len(ref),
                 checkpoint_sha256=checkpoint_hash,
                 parameters=sum(p.numel() for p in model.parameters()),registration_ms=enroll_ms, settings={})
    if args.target:
        target=read(args.target)
        results['mixture_si_sdr_db']=sisdr(x,target)
        results['output_si_sdr_db']=sisdr(ref,target)
    ref_times=[]
    for _ in range(args.trials):
        b=time.perf_counter()
        reference(model,x,enroll)
        ref_times.append((time.perf_counter()-b)*SR/len(x))
    results['whole_rtf_including_enrollment']=ref_times
    for size in args.chunks:
        # Warm the exact execution shape before paired timing.
        stream_run(model,emb,x[:min(len(x),max(2048,3*size))],size)
        speeds, service, first, paced=[],[],[],[]
        for _ in range(args.trials):
            y,rtf,ms,ft=stream_run(model,emb,x,size)
            speeds.append(rtf); service.extend(ms); first.append(ft)
            paced.append(paced_first(model,emb,x,size))
        delta=error(y,ref)
        sf.write(dest/f'stream_{size}.wav',y.numpy(),SR,subtype='FLOAT')
        results['settings'][size]=dict(**delta,rtf_trials=speeds,rtf_median=statistics.median(speeds),
            push_p50_ms=float(np.percentile(service,50)),push_p95_ms=float(np.percentile(service,95)),
            first_input_audio_ms=first[0][0],paced_first_ms=paced,
            first_wall_median_ms=statistics.median(paced))
        print(size,results['settings'][size],flush=True)
        if delta['max_abs']>2e-5 or delta['snr_db']<90:
            raise AssertionError(f"Streaming equivalence failed at {size}")
    # Same test recording, suffixes changed at four different positions / hop phases.
    tests=[]
    shortest=min(len(x),16000)
    probe=x[:shortest]
    offline=reference(model,probe,enroll)
    y0=stream_run(model,emb,probe,128)[0]
    rng=torch.Generator().manual_seed(2026)
    for cut in (shortest//3, shortest//2, shortest*2//3+17, shortest*3//4+127):
        changed=probe.clone()
        changed[cut:]=torch.randn(len(probe)-cut,generator=rng)*0.18
        # Last contributing frame for a hop [k,k+128) ends at k+512 (exclusive).
        prefix=max(0,((cut-512)//128 + 1)*128)
        y1=stream_run(model,emb,changed,128)[0]
        o1=reference(model,changed,enroll)
        err_s=(y0-y1).abs(); err_o=(offline-o1).abs()
        changed_indices=torch.nonzero(err_s>1e-7).flatten()
        first=changed_indices[0].item() if len(changed_indices) else None
        if first is None or first < prefix or err_o[:prefix].max().item()>2e-5:
            raise AssertionError("future suffix changed a finalized prefix")
        tests.append(dict(cut=cut,protected_prefix_samples=prefix,max_stream=float(err_s[:prefix].max()),
            max_official=float(err_o[:prefix].max()),first_change_at_1e_7=first,observed_future_samples=cut-first))
    results['suffix_probes']=tests
    (dest/'metrics.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
    print('Probes',tests,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mixture',type=Path,default=ROOT/'examples/mixture.wav')
    p.add_argument('--enroll',type=Path,default=ROOT/'examples/enroll.wav')
    p.add_argument('--target',type=Path)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'assets/spk_emb_causal_100')
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--chunks',nargs='+',type=int,default=[128,512,1024])
    p.add_argument('--trials',type=int,default=3)
    main(p.parse_args())
