# TSE · 目标说话人流式提取

使用 REAL-TSE 官方 `spk_emb_causal_100` 权重。输入为 16 kHz 单声道混合音频和目标人的注册录音，输出为可播放的 16 kHz 单声道目标波形。仓库默认用 CPU、float32。调研对比表在 `research/`；实测数据和输出在 `results/`。

## 本机运行

在仓库根目录的 PowerShell 窗口执行：

```powershell
.\.venv\Scripts\python.exe fetch_weights.py
.\.venv\Scripts\python.exe prepare_audio.py
.\.venv\Scripts\python.exe extract.py examples/mixture.wav examples/enroll.wav results/my_output.wav --chunk 128
.\.venv\Scripts\python.exe benchmark.py --target examples/target.wav --threads 2 --trials 3
```

本机独立虚拟环境复用工作区已有的 PyTorch 2.5.1 CPU。新机器可用 Python 3.12 创建环境，然后安装与平台匹配的 CPU 包：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe fetch_weights.py
```

`fetch_weights.py` 校验已存在的本地权重，首次下载约 1.05 GB 的官方 ZIP，并提取此模型的两个文件。完整模型含注册网络，共 25,049,608 个参数。首次运行 `prepare_audio.py` 会从 OpenSLR 下载 Mini LibriSpeech 开发集，按照已记录的来源生成样例。公开 PR 收录源码、调研、生成方法和测速数字。本机工作目录及本地 `archive/local-reproduction` 分支留存输入和输出 WAV。来源版本见 `third_party/UPSTREAM.md`。

## 分块接口

```python
import torch
from tse_model import load_model, enrollment
from streaming import Stream

with torch.inference_mode():
    model = load_model()
    stream = Stream(model, enrollment(model, registration_waveform))
    for pcm in incoming_16k_mono_float32_blocks:
        consume(stream.push(pcm))  # 暂无可提交输出时返回空张量
    consume(stream.flush())        # 只在这段录音结束时调用一次
```

每条正在处理的录音独享一个 Stream。注册端计算一次 ECAPA 表征。混合端缓存分析窗的重叠样本、6 层时序 LSTM 的 h/c、重叠相加分子与窗平方和。跨频带的双向 LSTM 每帧处理同一时刻的 32 个子带，无需未来语音。已处理的音频不会重新输入模型。

整段基线使用原始模型的 forward，保留官方 centered Hann STFT、两端 reflect padding 和 iSTFT 默认长度。关闭依赖整段峰值的后置增益。为保持相同注册，两条路径使用相同种子生成官方 Kaldi fbank 的 dither。分块支持不整除 hop 的输入，短于或等于 256 个样本的整条录音与官方 reflect 窗限制相同。输出长度为 `floor(N / 128) * 128`，示例最后不足一帧的 53 个采样也按官方行为截去。每次 push 可接任意正数个采样，块越大，真实回调积累等待越长。

## 延迟如何计算

16 kHz 下，窗长 512 样本（32 ms），hop 为 128 样本（8 ms）。模型无额外时序前视。按挑战赛口径，STFT/OLA 算法延迟为 384 样本（24 ms），输入按 hop 交付时另有 8 ms 帧缓冲。

对从 k 开始的一个输出 hop（k 是 128 的整数倍），在已读到混合信号区间 [0, k+512) 后即可提交整个 [k, k+128)。因此相对 hop 末端的等待为 384 样本；对任意单个输出样点的保守未来依赖上界为 511 样本（31.94 ms）。首个 128 个输出样本需要先收到 512 个输入样本。大于一个 hop 的应用回调会再引入调度等待。上述数字不含运算、音频驱动或设备缓冲。

基准测试会改变同一段录音四个位置以后的波形，检查理论上已可提交的全部输出是否保持不变，并要求阈值内确实检测到扰动。结构分析给出上界，扰动实验检验所选切点。分块与整段比较使用全部输出采样，包括开头和 flush 尾部，无移位搜索、裁边或分别归一化。

## 音频与测速口径

示例来自 OpenSLR 31 的 Mini LibriSpeech，注册语音和目标语音为说话人 1272 的不同句子，干扰人为 1462。保留了目标和干扰参考，可试听或算 SI-SDR。来源文件名与混合规则在 `examples/source.json`，有原数据归档时可运行 `prepare_audio.py` 重新生成。[数据来源](https://www.openslr.org/31/)

RTF 为不做音频节拍等待时的处理墙钟时间除以输入音频时长。流式 RTF 含缓存初始化、所有 push 和 flush，排除已预计算的注册向量及 WAV 读写。整段基线另列含注册的 RTF。首包墙钟时间单独用实时节拍投喂音频，实际计时至返回首个输出，排除模型加载与注册。P50/P95 为正常输出阶段的完整回调耗时。结果仅代表该机器和线程设置。详见 `results/metrics.json` 与 `results/实测.md`。

## 8 ms CPU 后端

保留 PyTorch 作为官方结构的对照。OpenVINO 直接从载入的 FP32 权重构建单帧图。ONNX Runtime 对分离器的矩阵乘和频带 LSTM 做动态 INT8 量化，注册、状态接口、STFT 和 OLA 均保留 FP32。两个后端都复用同一套波形缓存和 flush。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe profile_cpu.py
.\.venv\Scripts\python.exe extract.py examples/mixture.wav examples/enroll.wav results/ov.wav --backend openvino --threads 8 --chunk 128
.\.venv\Scripts\python.exe export_cpu.py
.\.venv\Scripts\python.exe extract.py examples/mixture.wav examples/enroll.wav results/int8.wav --backend ort-int8 --threads 2 --chunk 128
.\.venv\Scripts\python.exe benchmark_cpu.py --backend openvino --threads 8
.\.venv\Scripts\python.exe benchmark_cpu.py --backend ort-int8 --threads 2
```

INT8 默认每帧更新。可给最后一条命令增加 `--mask-period 4` 或 `--mask-period 8`，测量降低掩码更新频率的损失。该近似策略会跳过相应的循环状态更新，主要用于说明吞吐与质量的取舍。实测和其它可复现模型见[CPU 对比](results/CPU对比.md)与[替代模型](research/低延迟候选.md)。
