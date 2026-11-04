# 测试音频

来自 Mini LibriSpeech（OpenSLR 31），原语料为 LibriSpeech。Vassil Panayotov、Guoguo Chen、Daniel Povey、Sanjeev Khudanpur 等编制。原录音按 CC BY 4.0 许可使用。[Mini LibriSpeech](https://www.openslr.org/31/) · [LibriSpeech](https://www.openslr.org/12/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

选取不同句子作为注册和目标，另一个说话人作为干扰；经截取、等能量缩放并混合。确切文件名在 source.json，算法在仓库根目录的 prepare_audio.py。results 中的 WAV 是由该混合音频提取得到的派生文件。
