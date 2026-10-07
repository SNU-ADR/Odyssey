"""Turn off cuDNN benchmark mode inside traced TorchScript modules.

A traced graph stores each convolution as aten::_convolution with the cuDNN flags that were
set when it was traced (benchmark, deterministic, cudnn_enabled, allow_tf32) as constants, so
torch.backends.cudnn has no effect on it. The Cosmos tokenizer shipped with Fixer
(models/base/tokenizer_fast.pth) was traced with benchmark=True on all 81 convolutions of its
encoder and decoder. Benchmark mode times the cuDNN candidates on the first call per input
shape and keeps the fastest for the life of the process; when two candidates are close or the
GPU is shared, processes pick different ones and the restored images differ in the last bits
from the first frame on. With benchmark=False cuDNN uses its heuristic choice, which is the
same in every process on the same GPU and software.
"""
from __future__ import annotations

import torch

_BENCHMARK_INPUT = 9      # aten::_convolution(input, weight, bias, stride, padding, dilation,
                          #   transposed, output_padding, groups, benchmark, deterministic,
                          #   cudnn_enabled, allow_tf32)


def _script_graphs(module: torch.nn.Module):
    seen = set()
    for sub in module.modules():
        if isinstance(sub, torch.jit.ScriptModule) and id(sub) not in seen:
            seen.add(id(sub))
            for name in sub._c._method_names():
                yield sub._c._get_method(name).graph


def _benchmark_nodes(graph):
    def visit(block):
        for node in block.nodes():
            for inner in node.blocks():
                yield from visit(inner)
            if node.kind() == "aten::_convolution":
                flag = list(node.inputs())[_BENCHMARK_INPUT]
                if flag.node().kind() == "prim::Constant" and flag.toIValue() is True:
                    yield node
    yield from visit(graph.block())


def count_cudnn_benchmark(module: torch.nn.Module) -> int:
    """Convolutions in the module's scripted graphs that still run with benchmark=True."""
    return sum(1 for g in _script_graphs(module) for _ in _benchmark_nodes(g))


def disable_cudnn_benchmark(module: torch.nn.Module) -> int:
    """Rewrite benchmark=True to False in every scripted graph of `module`, before its first call.

    Returns the number of convolutions changed; raises if any remain afterwards.
    """
    changed = 0
    for graph in _script_graphs(module):
        nodes = list(_benchmark_nodes(graph))
        if not nodes:
            continue
        false = graph.insertConstant(False)
        false.node().moveBefore(next(iter(graph.nodes())))
        for node in nodes:
            node.replaceInput(_BENCHMARK_INPUT, false)
        torch._C._jit_pass_lint(graph)
        changed += len(nodes)
    remaining = count_cudnn_benchmark(module)
    if remaining:
        raise RuntimeError("%d scripted convolutions still have cudnn benchmark=True" % remaining)
    return changed
