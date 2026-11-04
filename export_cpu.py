"""Export the official separator as a fixed one-hop ONNX graph with explicit h/c."""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch

from tse_model import ROOT, load_model


class Graph:
    def __init__(self):
        self.nodes = []
        self.constants = []

    def tensor(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        name = f"w{len(self.constants)}"
        self.constants.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def op(self, kind, *inputs, **attrs):
        name = f"{kind}_{len(self.nodes)}"
        self.nodes.append(helper.make_node(kind, inputs, [name], name=name, **attrs))
        return name

    def reshape(self, x, shape):
        return self.op("Reshape", x, self.tensor(np.array(shape, np.int64)))

    def split(self, x, widths, axis):
        names = [f"split_{len(self.nodes)}_{i}" for i in range(len(widths))]
        self.nodes.append(helper.make_node("Split", [x, self.tensor(np.array(widths, np.int64))], names, axis=axis))
        return names

    def linear(self, x, module):
        w = module.weight.detach().reshape(module.weight.shape[:2]).T.contiguous()
        return self.op("Add", self.op("MatMul", x, self.tensor(w)), self.tensor(module.bias))

    def norm(self, x, module):
        return self.op("LayerNormalization", x, self.tensor(module.weight), self.tensor(module.bias),
                       axis=-1, epsilon=float(module.eps))


def port(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def export(model, dest):
    g = Graph()
    sep = model.sep_model
    widths = sep.band_split.band_width
    bands, dim = len(widths), sep.feature_dim
    inputs = [port("spectrum", [2, sum(widths)]), port("gate", [1, dim])]
    outputs = []
    features = []
    for raw, width, layer in zip(g.split("spectrum", widths, 1), widths, sep.subband_norm.BN):
        features.append(g.linear(g.norm(g.reshape(raw, [1, 2*width]), layer[0]), layer[1]))
    x = g.op("Mul", g.op("Concat", *features, axis=0), "gate")
    for i, block in enumerate(sep.separator.separation):
        time, freq = block.band_rnn, block.band_comm
        rnn = time.rnn
        hidden = rnn.hidden_size
        h, c = f"h{i}", f"c{i}"
        inputs.extend([port(h, [bands, hidden]), port(c, [bands, hidden])])
        def mm(x, w):
            return g.op("MatMul", x, g.tensor(w.T.contiguous()))
        gates = g.op("Add", g.op("Add", mm(g.norm(x, time.norm), rnn.weight_ih_l0),
                                 mm(h, rnn.weight_hh_l0)), g.tensor(rnn.bias_hh_l0+rnn.bias_ih_l0))
        inn, forget, cand, out = g.split(gates, [hidden]*4, 1)
        nc = g.op("Add", g.op("Mul", g.op("Sigmoid", forget), c),
                  g.op("Mul", g.op("Sigmoid", inn), g.op("Tanh", cand)))
        nh = g.op("Mul", g.op("Sigmoid", out), g.op("Tanh", nc))
        outputs.extend([port(nh, [bands, hidden]), port(nc, [bands, hidden])])
        x = g.op("Add", x, g.linear(nh, time.proj))
        # GroupNorm(1, dim) over the entire frequency sequence of this frame.
        normalized = g.op("LayerNormalization", g.reshape(x, [1, bands*dim]),
                          g.tensor(np.ones(bands*dim, np.float32)), axis=-1, epsilon=freq.norm.eps)
        z = g.op("Add", g.op("Mul", g.reshape(normalized, [bands, dim]),
                              g.tensor(freq.norm.weight)), g.tensor(freq.norm.bias))
        # PyTorch IFGO -> ONNX IOFC order.
        def pack(param, suffix):
            return torch.cat([getattr(freq.rnn, param+suffix).chunk(4)[k] for k in (0,3,1,2)])
        def weights(param):
            return g.tensor(torch.stack([pack(param, s) for s in ("", "_reverse")]))
        bias = g.tensor(torch.stack([torch.cat([pack("bias_ih_l0", s), pack("bias_hh_l0", s)])
                                     for s in ("", "_reverse")]))
        y = g.op("LSTM", g.reshape(z, [bands,1,dim]), weights("weight_ih_l0"),
                 weights("weight_hh_l0"), bias, hidden_size=hidden, direction="bidirectional")
        x = g.op("Add", x, g.linear(g.reshape(y, [bands, 2*hidden]), freq.proj))
    real, imag = [], []
    for z, width, head in zip(g.split(x, [1]*bands, 0), widths, sep.band_masker.mask):
        z = g.norm(z, head[0])
        for layer in (head[1], head[3]):
            z = g.op("Tanh", g.linear(z, layer))
        a,b,c,d = g.split(g.reshape(g.linear(z, head[5]), [4,width]), [1]*4, 0)
        real.append(g.op("Mul", a, g.op("Sigmoid", c)))
        imag.append(g.op("Mul", b, g.op("Sigmoid", d)))
    mask = g.op("Concat", g.op("Concat", *real, axis=1), g.op("Concat", *imag, axis=1), axis=0)
    net = helper.make_graph(g.nodes, "spk_emb_causal_100_one_hop", inputs,
                            [port(mask, [2,sum(widths)]), *outputs], g.constants)
    graph = helper.make_model(net, opset_imports=[helper.make_opsetid("",17)], ir_version=9)
    onnx.checker.check_model(graph)
    onnx.save(graph, dest)


if __name__ == "__main__":
    from onnxruntime.quantization import quantize_dynamic, QuantType
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=ROOT/"assets/onnx")
    p.add_argument("--checkpoint", type=Path, default=ROOT/"assets/spk_emb_causal_100")
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    fp = args.directory/"one_hop.onnx"
    export(load_model(args.checkpoint), fp)
    int8 = args.directory/"one_hop_int8.onnx"
    quantize_dynamic(fp, int8, op_types_to_quantize=["MatMul","LSTM"],
                     weight_type=QuantType.QInt8, per_channel=True)

    import json
    from ort_engine import digest
    manifest = {"graph_sha256": digest(int8), "checkpoint_sha256": digest(args.checkpoint/"avg_model.pt")}
    int8.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
