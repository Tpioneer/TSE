"""Build a disjoint enrollment and an equal-energy two-reader mixture from OpenSLR 31."""
from pathlib import Path
import io
import json
import tarfile
import urllib.request

import numpy as np
import soundfile as sf
from tse_model import ROOT


def prepare():
    dest = ROOT / "examples"
    dest.mkdir(exist_ok=True)
    corpus = ROOT / "assets/dev-clean-2.tar.gz"
    if not corpus.is_file():
        corpus.parent.mkdir(exist_ok=True)
        part = corpus.with_suffix(".part")
        urllib.request.urlretrieve("https://www.openslr.org/resources/31/dev-clean-2.tar.gz", part)
        part.replace(corpus)
    with tarfile.open(corpus, "r:gz") as archive:
        readers = {}
        for item in sorted(archive.getmembers(), key=lambda e: e.name):
            if item.name.endswith(".flac"):
                readers.setdefault(Path(item.name).stem.split("-")[0], []).append(item)
        first, second = list(readers.values())[:2]
        items = [first[0], first[1], second[0]]
        audio = []
        for item in items:
            data, sr = sf.read(io.BytesIO(archive.extractfile(item).read()), dtype="float32")
            assert sr == 16000
            audio.append(data)
    n = min(4*sr+53, len(audio[1]), len(audio[2]))
    ref = audio[0][:5*sr]
    target, other = audio[1][:n], audio[2][:n]
    other = other * (np.linalg.norm(target) / np.linalg.norm(other))
    gain = 0.85 / np.max(np.abs(target + other))
    for name, samples in [('enroll',ref),('target',target*gain),('interferer',other*gain),('mixture',(target+other)*gain)]:
        sf.write(dest / (name+'.wav'),samples,sr,subtype='FLOAT')
    meta = dict(source="https://www.openslr.org/31/",license="CC BY 4.0 (LibriSpeech)",files=[i.name for i in items],samples=n,sample_rate=sr,sir_db=0)
    (dest/'source.json').write_text(json.dumps(meta,indent=2),encoding='utf8')


if __name__ == '__main__':
    prepare()
