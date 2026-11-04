"""Test open-voicefilter-lite 0.2.0 on the same mono pair at its native 10 ms hop."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time

import numpy as np
import torch
from voicefilter_lite import StreamingVoiceFilter

from benchmark import SR, read, sisdr
from tse_model import ROOT


def main():
    torch.set_num_threads(1)
    x, enrollment, target = [read(ROOT / f"examples/{name}.wav") for name in ("mixture","enroll","target")]
    engine = StreamingVoiceFilter()
    engine.set_enrollment(enrollment.numpy())
    hop = engine.hop_length
    # OLA returns samples starting at the start of the newest FFT window.
    delay = engine.hp.n_fft - hop
    padded = np.pad(x.numpy(), (0, (-len(x)-delay)%hop+delay))
    def run():
        engine.reset_state()
        start = time.perf_counter()
        out = np.concatenate([engine.step(b) for b in padded.reshape(-1,hop)])
        return torch.from_numpy(out[delay:delay+len(x)]), (time.perf_counter()-start)*SR/len(x)
    run()
    rates = []
    for _ in range(3):
        y, rtf = run()
        rates.append(rtf)
    score = sisdr(y,target)
    report = dict(package="open-voicefilter-lite==0.2.0", sample_rate=SR, input_samples=len(x),
                  fft=engine.hp.n_fft, window=engine.hp.win_length, hop=hop,
                  alignment_delay_samples=delay, alignment_delay_ms=delay*1000/SR,
                  rtf_trials=rates, rtf_median=float(np.median(rates)),
                  target_si_sdr_db=score, mixture_si_sdr_db=sisdr(x,target),
                  si_sdri_db=score-sisdr(x,target))
    print(report,flush=True)
    (ROOT/"results/voicefilter_lite.json").write_text(json.dumps(report,indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
