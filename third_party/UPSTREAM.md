# Inference sources

The model code was copied from these public upstream snapshots:

- WeSep REAL-TSE `2a540977a348fbaa92e623210505430e2cec608d`: [source tree](https://github.com/REAL-TSE/wesep-real-tse/tree/2a540977a348fbaa92e623210505430e2cec608d). The selected files under `wesep/` are the upstream model, feature, normalization, fusion and registration modules as published. Their original notices are preserved.
- WeSpeaker `8f53b6485d9f88a207bd17e7f8dba899495ec794`: [source tree](https://github.com/wenet-e2e/wespeaker/tree/8f53b6485d9f88a207bd17e7f8dba899495ec794). The ECAPA and pooling files are unchanged. The local speaker_model.py registry exposes only the published checkpoint's encoder. Apache-2.0 license is included.

The inference package excludes optional upstream CLI, VAD, training and unrelated encoder imports. Whole-waveform forward uses the upstream TSE_BSRNN_SPK class with strict state loading. The local fetch script verifies the individual checkpoint and configuration against SHA-256. Full source archives, the upstream model archive, extracted checkpoint and Mini LibriSpeech archive are stored locally under assets/ or vendor/ and excluded from the Git history.
