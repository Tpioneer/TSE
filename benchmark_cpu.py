"""Compare one-hop CPU backends to the official waveform on the PR fixture."""
import argparse
import hashlib
import json
import platform
import statistics
import time

import numpy as np
import torch

from benchmark import SR, error, read, reference, sisdr
from tse_model import ROOT, enrollment, load_model


def run(engine, emb, audio, first_only=False):
    t0 = time.perf_counter()
    stream = engine.stream(emb)
    outputs, times = [], []
    first_samples = None
    for offset in range(0, len(audio), 128):
        block = audio[offset:offset+128]
        if first_only:
            wait = t0 + (offset+len(block))/SR - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        start = time.perf_counter()
        y = stream.push(block)
        end = time.perf_counter()
        if y.numel():
            if first_samples is None:
                first_samples = offset+len(block)
                if first_only:
                    assert first_samples == 512
                    return 1000*(end-t0)
            if len(block)==128:
                times.append(1000*(end-start))
        outputs.append(y)
    outputs.append(stream.flush())
    y = torch.cat(outputs)
    return y, (time.perf_counter()-t0)*SR/len(audio), times


def probes(engine, emb, x):
    x = x[:16000]
    baseline = run(engine, emb, x)[0]
    rng = torch.Generator().manual_seed(2026)
    tests = []
    for cut in (5333,8000,10683,12127):
        suffix = x.clone()
        suffix[cut:] = torch.randn(len(x)-cut,generator=rng)*0.18
        diff = (run(engine, emb, suffix)[0]-baseline).abs()
        protected = ((cut-512)//128+1)*128
        change = torch.nonzero(diff>1e-7).flatten()
        assert len(change) and int(change[0]) >= protected and float(diff[:protected].max()) == 0
        tests.append(dict(cut=cut,protected=protected,max_abs=float(diff[:protected].max()),
                          first_changed=int(change[0])))
    return tests


@torch.inference_mode()
def main(args):
    if args.trials < 1 or args.threads < 1 or (args.backend == "openvino" and args.mask_period != 1):
        raise ValueError("Invalid trial, thread, or mask-period settings")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    x, e, target = [read(ROOT/f"examples/{name}.wav") for name in ("mixture","enroll","target")]
    model = load_model()
    ref = reference(model,x,e)
    emb = enrollment(model,e)
    before = time.perf_counter()
    if args.backend == "openvino":
        from cpu_engine import CpuEngine
        engine = CpuEngine(model,args.threads)
    else:
        from ort_engine import OrtEngine, verify_graph
        graph = ROOT/"assets/onnx/one_hop_int8.onnx"
        verify_graph(graph, ROOT/"assets/spk_emb_causal_100")
        before = time.perf_counter()
        engine = OrtEngine(model,graph,args.threads,args.mask_period)
    setup_ms = (time.perf_counter()-before)*1000
    run(engine,emb,x[:4096])
    rates, calls, first = [], [], []
    for _ in range(args.trials):
        y,rtf,times = run(engine,emb,x)
        rates.append(rtf)
        calls.extend(times)
        first.append(run(engine,emb,x,True))
        print(args.backend, "RTF",rtf,"paced first ms",first[-1],flush=True)
    delta = error(y,ref)
    score = sisdr(y,target)
    official = sisdr(ref,target)
    assert y.shape == ref.shape and torch.isfinite(y).all()
    if args.backend == "openvino":
        assert delta["max_abs"] < 2e-5 and delta["snr_db"] > 90
    report = dict(backend=args.backend,mask_period=args.mask_period,threads=args.threads, processor=platform.processor(),
                  input_samples=len(x),first_input_samples=512,backend_setup_ms=setup_ms,
                  mixture_pcm_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                  **delta, target_si_sdr_db=score, official_si_sdr_db=official, delta_si_sdr_db=score-official,
                  rtf_trials=rates,rtf_median=statistics.median(rates),paced_first_ms=first,
                  first_wall_median_ms=statistics.median(first),push_p50_ms=float(np.percentile(calls,50)),
                  push_p95_ms=float(np.percentile(calls,95)),suffix_probes=probes(engine,emb,x))
    (ROOT/f"results/cpu_{args.backend}_k{args.mask_period}.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(report, flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend",choices=["openvino","ort-int8"],required=True)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--mask-period",type=int,default=1)
    p.add_argument("--trials",type=int,default=3)
    main(p.parse_args())
