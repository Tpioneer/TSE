"""Stateful CPU/float32 waveform inference for the official spk_emb_causal_100 model."""
import torch


class Stream:
    """One utterance and one precomputed target embedding. push() returns finalized samples."""

    def __init__(self, model, embedding):
        self.sep = model.sep_model
        self.fuse = model.spk_ft.spkemb
        self.embedding = embedding
        self.win, self.hop = self.sep.win, self.sep.stride
        self.half = self.win // 2
        self.window = torch.hann_window(self.win)
        self.states = [None] * len(self.sep.separator.separation)
        self.buffer = torch.empty(0)
        self.tail = torch.empty(0)
        self.ola = torch.zeros(self.win)
        self.weight = torch.zeros(self.win)
        self.skip = self.half
        self.received = self.emitted = 0
        self.started = self.closed = False

    def _mask(self, spec):
        bands = self.sep.band_split
        feats = self.sep.subband_norm(bands(torch.stack((spec.real, spec.imag), 1)))
        feats = self.fuse.post(feats, self.embedding)
        _, bands_n, dim, steps = feats.shape
        for i, block in enumerate(self.sep.separator.separation):
            time = block.band_rnn
            x = feats.reshape(bands_n, dim, steps)
            y, self.states[i] = time.rnn(time.norm(x).transpose(1, 2).contiguous(), self.states[i])
            x = x + time.proj(y).transpose(1, 2)
            # Bidirectional along frequency, independently for each time frame.
            x = x.permute(2, 1, 0).contiguous()
            feats = block.band_comm(x).permute(2, 1, 0).contiguous().unsqueeze(0)
        ri = self.sep.band_masker(feats, bands(spec))
        return torch.complex(ri[:, 0], ri[:, 1])[0, 0].T

    def _frames(self):
        if self.buffer.numel() < self.win:
            return torch.empty(0)
        frames = self.buffer.unfold(0, self.win, self.hop)
        count = frames.shape[0]
        spec = torch.fft.rfft(frames * self.window).T.unsqueeze(0)
        frames_out = torch.fft.irfft(self._mask(spec), n=self.win) * self.window
        self.buffer = self.buffer[count*self.hop:].clone()
        chunks = []
        for frame in frames_out:
            self.ola += frame
            self.weight += self.window.square()
            chunks.append(self.ola[:self.hop] / self.weight[:self.hop].clamp_min(1e-12))
            self.ola = torch.cat((self.ola[self.hop:], self.ola.new_zeros(self.hop)))
            self.weight = torch.cat((self.weight[self.hop:], self.weight.new_zeros(self.hop)))
        out = torch.cat(chunks)
        trim = min(self.skip, out.numel())
        self.skip -= trim
        return out[trim:]

    @torch.inference_mode()
    def push(self, x):
        if self.closed:
            raise RuntimeError("stream already flushed")
        if x.ndim != 1 or x.dtype != torch.float32 or x.device.type != 'cpu':
            raise ValueError("expected mono float32 CPU samples at 16 kHz")
        self.received += x.numel()
        self.tail = torch.cat((self.tail, x))[-self.half-1:].clone()
        self.buffer = torch.cat((self.buffer, x))
        if not self.started:
            if self.received <= self.half:
                return torch.empty(0)
            self.buffer = torch.cat((self.buffer[1:self.half+1].flip([0]), self.buffer))
            self.started = True
        out = self._frames()
        self.emitted += out.numel()
        return out

    @torch.inference_mode()
    def flush(self):
        if self.closed:
            raise RuntimeError("stream already flushed")
        if not self.started:
            raise ValueError("official centered reflect STFT needs more than 256 samples")
        self.closed = True
        self.buffer = torch.cat((self.buffer, self.tail[:-1].flip([0])))
        out = self._frames()
        # torch.istft(length=None) truncates the trailing fractional hop.
        need = (self.received // self.hop)*self.hop - self.emitted - out.numel()
        if not 0 <= need <= self.win-self.hop:
            raise RuntimeError("inconsistent overlap-add tail")
        tail = self.ola[:need] / self.weight[:need].clamp_min(1e-12)
        self.emitted += out.numel() + need
        return torch.cat((out, tail))
