"""Fixed one-hop OpenVINO CPU graph for the released 32-band BSRNN checkpoint."""
import numpy as np
import openvino as ov
from openvino import opset13 as op
import torch

from streaming import Stream


def array(t):
    return t.detach().cpu().numpy()


def reshape(x, shape):
    return op.reshape(x, np.asarray(shape, dtype=np.int64), False)


def linear(x, module):
    w = array(module.weight)
    return op.add(op.matmul(x, w.reshape(w.shape[:2]), False, True), array(module.bias))


def norm(x, module, axes):
    y = op.mvn(x, op.constant(np.array(axes, np.int64)), True, module.eps, "inside_sqrt")
    return op.add(op.multiply(y, array(module.weight)), array(module.bias))


def band_lstm(x, module, bands, dim):
    rnn = module.rnn
    # OpenVINO LSTMSequence consumes FICO, PyTorch stores IFGO.
    def pack(attr):
        return np.stack([np.concatenate([np.split(array(getattr(rnn, attr+suffix)), 4)[g]
                                         for g in (1,0,2,3)]) for suffix in ("","_reverse")])
    hidden = rnn.hidden_size
    zero = np.zeros((1,2,hidden),np.float32)
    seq = op.lstm_sequence(reshape(norm(x,module.norm,[0,1]),[1,bands,dim]),
                           zero,zero,np.array([bands],np.int32),
                           pack("weight_ih_l0"),pack("weight_hh_l0"),
                           pack("bias_ih_l0")+pack("bias_hh_l0"),hidden,"bidirectional")
    y = reshape(op.transpose(seq.output(0),np.array([0,2,1,3],np.int64)),[bands,2*hidden])
    return op.add(x,linear(y,module.proj))


def build_graph(model):
    sep = model.sep_model
    widths = sep.band_split.band_width
    bands, dim = len(widths), sep.feature_dim
    spec = op.parameter([2,sum(widths)],np.float32,name="spectrum")
    gate = op.parameter([1,dim],np.float32,name="registration_gate")
    splits = op.variadic_split(spec,np.int64(1),np.array(widths,np.int64))
    features = []
    for i, layer in enumerate(sep.subband_norm.BN):
        x = reshape(splits.output(i),[1,2*widths[i]])
        features.append(linear(norm(x,layer[0],[1]),layer[1]))
    x = op.multiply(op.concat(features,0),gate)
    states, updated = [], []
    for i, block in enumerate(sep.separator.separation):
        time = block.band_rnn
        hidden = time.rnn.hidden_size
        h = op.parameter([bands,hidden],np.float32,name=f"h{i}")
        c = op.parameter([bands,hidden],np.float32,name=f"c{i}")
        states.extend([h,c])
        rnn=time.rnn
        gates = op.add(op.add(op.matmul(norm(x,time.norm,[1]),array(rnn.weight_ih_l0),False,True),
                              op.matmul(h,array(rnn.weight_hh_l0),False,True)),
                       array(rnn.bias_ih_l0+rnn.bias_hh_l0))
        parts=op.split(gates,np.int64(1),4)
        new_c=op.add(op.multiply(op.sigmoid(parts.output(1)),c),
                     op.multiply(op.sigmoid(parts.output(0)),op.tanh(parts.output(2))))
        new_h=op.multiply(op.sigmoid(parts.output(3)),op.tanh(new_c))
        updated.extend([new_h,new_c])
        x=band_lstm(op.add(x,linear(new_h,time.proj)),block.band_comm,bands,dim)
    band_x=op.split(x,np.int64(0),bands)
    real, imag = [], []
    for i, head in enumerate(sep.band_masker.mask):
        y=norm(band_x.output(i),head[0],[1])
        y=op.tanh(linear(y,head[1]))
        y=op.tanh(linear(y,head[3]))
        y=reshape(linear(y,head[5]),[4,widths[i]])
        parts=op.split(y,np.int64(0),4)
        real.append(op.multiply(parts.output(0),op.sigmoid(parts.output(2))))
        imag.append(op.multiply(parts.output(1),op.sigmoid(parts.output(3))))
    mask=op.concat([op.concat(real,1),op.concat(imag,1)],0)
    return ov.Model([mask,*updated],[spec,gate,*states],"one_hop_spk_emb_causal_100")


class CpuEngine:
    """Immutable compiled weights. Each open stream owns a separate request and h/c."""
    def __init__(self, model, threads=4):
        self.model=model
        self.compiled=ov.Core().compile_model(build_graph(model),"CPU",
            {"INFERENCE_NUM_THREADS":threads,"NUM_STREAMS":1,
             "PERFORMANCE_HINT":"LATENCY","INFERENCE_PRECISION_HINT":"f32"})

    def stream(self, embedding):
        return CpuStream(self,embedding)


class CpuStream(Stream):
    def __init__(self, engine, embedding):
        super().__init__(engine.model,embedding)
        self.request=engine.compiled.create_infer_request()
        self.gate=self.fuse.post(torch.ones(1,1,self.sep.feature_dim,1),embedding).detach().numpy().reshape(1,-1)
        self.cache=[np.zeros(tuple(port.shape),np.float32) for port in engine.compiled.inputs[2:]]

    def _mask(self, spec):
        outputs=[]
        for frame in spec[0].T:
            ri=torch.stack((frame.real,frame.imag)).contiguous().numpy()
            self.request.infer([ri,self.gate,*self.cache],share_inputs=True,share_outputs=True)
            mask=self.request.get_output_tensor(0).data
            outputs.append(frame * torch.from_numpy(mask[0]+1j*mask[1]))
            for i, state in enumerate(self.cache):
                np.copyto(state,self.request.get_output_tensor(i+1).data)
        return torch.stack(outputs)
