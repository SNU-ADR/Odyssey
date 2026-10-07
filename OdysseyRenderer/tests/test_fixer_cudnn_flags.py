import importlib.util
import io
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]          # OdysseyRenderer/


def load_cudnn_flags():
    spec = importlib.util.spec_from_file_location("cudnn_flags", ROOT / "fixer" / "cudnn_flags.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.b = torch.nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


def traced(benchmark):
    """A traced module saved and reloaded, as the tokenizer is shipped (torch.jit.load of bytes)."""
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = benchmark
    try:
        module = torch.jit.trace(Block().eval(), torch.zeros(1, 3, 8, 8))
    finally:
        torch.backends.cudnn.benchmark = previous
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)


class Holder(torch.nn.Module):
    """Like Pix2Pix_Turbo: scripted encoder/decoder registered under a plain module."""

    def __init__(self):
        super().__init__()
        self.encoder = traced(True)
        self.decoder = traced(True)
        self.head = torch.nn.Conv2d(2, 2, 1)


def test_trace_bakes_benchmark_flag():
    flags = load_cudnn_flags()
    assert flags.count_cudnn_benchmark(traced(True)) == 2
    assert flags.count_cudnn_benchmark(traced(False)) == 0


def test_disable_rewrites_every_scripted_convolution_and_keeps_outputs():
    flags = load_cudnn_flags()
    torch.manual_seed(0)
    reference = Holder().eval()
    torch.manual_seed(0)
    holder = Holder().eval()                                  # same weights, rewritten before any call
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        before = reference.decoder(torch.zeros(2, 3, 8, 8)), reference.encoder(x)
    assert flags.count_cudnn_benchmark(holder) == 4
    assert flags.disable_cudnn_benchmark(holder) == 4
    assert flags.count_cudnn_benchmark(holder) == 0
    with torch.no_grad():
        after = holder.decoder(torch.zeros(2, 3, 8, 8)), holder.encoder(x)
    assert all(torch.equal(b, a) for b, a in zip(before, after))
    assert flags.disable_cudnn_benchmark(holder) == 0          # idempotent


def test_rewritten_graph_survives_save_and_load():
    flags = load_cudnn_flags()
    module = traced(True)
    flags.disable_cudnn_benchmark(module)
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    buffer.seek(0)
    assert flags.count_cudnn_benchmark(torch.jit.load(buffer)) == 0
