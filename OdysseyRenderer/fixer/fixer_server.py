"""Fixer restoration server (sidecar) for the simulator.

Adapted from the inference code of NVIDIA Fixer, by way of a scene viewer's Fixer
server. It keeps that server's wire format, resampling and timestep, so a simulator
frame restored here matches what the viewer shows for the same checkpoint.

Why a separate process: Fixer needs cosmos-predict2 on python 3.12 / torch 2.8,
the renderer runs in the simulator interpreter (python 3.9). The two do not merge, so the
simulator talks to this over HTTP (restorer_client.HttpRestorer). The simulator
normally starts and stops this itself (ODYSSEY_RESTORER=fixer_h1b16 | fixer_pretrained);
running it by hand is only for debugging.

  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=<fixer>/shims FIXER_MODELS_DIR=<fixer>/models \
  <fixer-env>/bin/python fixer_server.py --port 18100 \
      --checkpoint <fixer>/models/pretrained/pretrained_fixer.pkl --src-dir <fixer>/src

Changes from the viewer copy: /health also reports the checkpoint path and a
label, so the caller can refuse to run on the wrong weights, and the default
host is 127.0.0.1.

Wire format: [4-byte big-endian JSON header length][JSON {"names","shape","dtype"}]
[contiguous uint8 (N,H,W,3) RGB]. A failed restore is an error, never the input.

The header may add "channel_order": "bgr" (advertised in /health as channel_orders). The
batch is then BGR both ways and the channel flip happens on the GPU. The simulator renders
BGR, and reversing a 25 MB batch's channel axis on the CPU was two strided copies per step
(35 + 23 ms at 4x1080p) against a free flip on the device. Without the key it is RGB, as the
viewer sends it.

ODYSSEY_FIXER_EMPTY_CACHE=0 keeps the CUDA allocator cache between requests. The default (1)
empties it after every request, as the viewer copy does: there several model servers share
one GPU between sporadic clicks. A rollout sends one fixed-shape batch every step, and
re-allocating after each empty cost +25 ms mean and a 150 ms p90 per request on a B200,
with identical output.

The Cosmos tokenizer's traced encoder/decoder carry cudnn benchmark=True on every
convolution; Restorer rewrites it to False before the first request (cudnn_flags.py), so
every process restores the same input to the same bytes. This costs about 10% Fixer time.

"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback

import numpy as np
import torch
from flask import Flask, Response, jsonify, request

from cudnn_flags import disable_cudnn_benchmark

EMPTY_CACHE = os.environ.get("ODYSSEY_FIXER_EMPTY_CACHE", "1").strip() != "0"
CHANNEL_ORDERS = ("rgb", "bgr")


def pack_batch(names, batch):
    head = json.dumps({"names": list(names), "shape": list(batch.shape),
                       "dtype": str(batch.dtype)}).encode()
    return len(head).to_bytes(4, "big") + head + np.ascontiguousarray(batch).tobytes()


def unpack_batch(blob):
    n = int.from_bytes(blob[:4], "big")
    head = json.loads(blob[4:4 + n].decode())
    arr = np.frombuffer(blob[4 + n:], dtype=np.dtype(head["dtype"])).reshape(head["shape"])
    return head["names"], arr, head.get("channel_order", "rgb")


PROC_H, PROC_W = 576, 1024          # Fixer's training resolution
TIMESTEP = 250                      # Fixer's default


class Restorer:
    def __init__(self, checkpoint, src_dir, device="cuda", dtype=torch.bfloat16):
        if src_dir and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import Pix2Pix_Turbo

        self.device = device
        self.dtype = dtype
        net = Pix2Pix_Turbo(pretrained_path=checkpoint, timestep=TIMESTEP)
        net.set_eval()
        self.net = net.to(device=device, dtype=dtype)
        # The tokenizer's traced encoder/decoder run their convolutions with cudnn benchmark=True
        # baked in, so the restored image depended on which process timed which kernel fastest.
        self.cudnn_benchmark_disabled = disable_cudnn_benchmark(self.net)
        print("[fixer] cudnn benchmark disabled on %d scripted convolutions"
              % self.cudnn_benchmark_disabled, flush=True)
        self.lock = threading.Lock()
        self._warmed = set()
        # Pix2Pix_Turbo bakes the batch size into `timesteps` and `condition`
        # (padding_mask is (B,1,H,W)) at construction. Rebuild per batch size.
        self._batch_state: dict[int, tuple] = {}
        self._capture_batch_state(1)

    def _capture_batch_state(self, n: int):
        if n in self._batch_state:
            return self._batch_state[n]
        from cosmos_predict2.conditioner import DataType

        net = self.net
        prev = net.batch_size
        net.batch_size = n
        timesteps = torch.tensor([TIMESTEP], device=self.device).repeat(n)
        _, uncondition = net.unet.conditioner.get_condition_uncondition(net.sample_batch_image())
        condition = uncondition.edit_data_type(DataType.IMAGE)
        net.batch_size = prev
        self._batch_state[n] = (timesteps, condition)
        return self._batch_state[n]

    def _use_batch(self, n: int):
        timesteps, condition = self._capture_batch_state(n)
        self.net.batch_size = n
        self.net.timesteps = timesteps
        self.net.condition = condition

    @staticmethod
    def _resize(x, size):
        import torch.nn.functional as F
        return F.interpolate(x.float(), size=size, mode="bilinear",
                             align_corners=False, antialias=True).clamp_(0.0, 1.0)

    @torch.no_grad()
    def _run(self, img_t01):
        n = int(img_t01.shape[0])
        self._use_batch(n)
        x = (img_t01 * 2.0 - 1.0).to(self.dtype)
        with torch.autocast("cuda", dtype=self.dtype, enabled=True):
            out = self.net(x).float()
        return (out * 0.5 + 0.5).clamp_(0.0, 1.0)

    @torch.no_grad()
    def restore(self, batch_uint8, bgr=False):
        """(N,H,W,3) uint8 -> (N,H,W,3) uint8, same size and channel order in and out.

        The network runs on RGB; bgr=True flips the channels on the device both ways.
        """
        t = torch.from_numpy(batch_uint8).to(self.device)
        if bgr:
            t = t.flip(-1)
        t = t.permute(0, 3, 1, 2).float().div_(255.0)
        h, w = int(t.shape[-2]), int(t.shape[-1])
        proc = self._resize(t, (PROC_H, PROC_W))
        n = int(proc.shape[0])
        if n not in self._warmed:
            self._run(proc)          # first pass at a new batch size is the slow one
            self._warmed.add(n)
        out = self._resize(self._run(proc), (h, w))
        out = (out.clamp_(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)
        if bgr:
            out = out.flip(-1)
        # Contiguous on the device, so pack_batch's ascontiguousarray is not a strided CPU copy.
        return out.contiguous().cpu().numpy()


def build_app(state):
    app = Flask(__name__)

    @app.get("/health")
    def health():
        ready = state["restorer"] is not None
        vram = None
        gpu = None
        if torch.cuda.is_available():
            vram = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
            gpu = torch.cuda.get_device_name(0)
        return jsonify({
            "ready": ready,
            "model": "fixer",
            "label": state["label"],
            "checkpoint": state["checkpoint"],
            "error": state["error"],
            "proc_hw": [PROC_H, PROC_W],
            "timestep": TIMESTEP,
            "gpu": gpu,
            "vram_mib": vram,
            "pid": os.getpid(),
            "channel_orders": list(CHANNEL_ORDERS),
            "empty_cache": EMPTY_CACHE,
            "cudnn_benchmark_disabled": getattr(state["restorer"], "cudnn_benchmark_disabled", None),
        })

    @app.post("/restore")
    def restore():
        r = state["restorer"]
        if r is None:
            return jsonify({"error": state["error"] or "restorer not loaded"}), 503
        blob = request.get_data(cache=False)
        if not blob:
            return jsonify({"error": "no images"}), 400
        try:
            names, batch, order = unpack_batch(blob)
        except Exception as exc:                                    # noqa: BLE001
            return jsonify({"error": "malformed batch: %s" % exc}), 400
        if batch.ndim != 4 or batch.shape[-1] != 3 or batch.dtype != np.uint8:
            return jsonify({"error": "expected a uint8 (N,H,W,3) batch, got %s %s"
                                     % (batch.shape, batch.dtype)}), 400
        if order not in CHANNEL_ORDERS:
            return jsonify({"error": "channel_order must be one of %s, got %r"
                                     % (CHANNEL_ORDERS, order)}), 400

        t0 = time.time()
        with r.lock:                                                # one GPU, one model
            out = r.restore(np.ascontiguousarray(batch), bgr=(order == "bgr"))
            if EMPTY_CACHE:
                torch.cuda.empty_cache()
        elapsed = round((time.time() - t0) * 1000.0, 1)

        # channel_order is echoed so a client can refuse a server that ignored it.
        meta = {"n": len(names), "ms": elapsed, "proc_hw": [PROC_H, PROC_W],
                "timestep": TIMESTEP, "channel_order": order}
        resp = Response(pack_batch(names, out), mimetype="application/octet-stream")
        resp.headers["X-Fixer-Meta"] = json.dumps(meta)
        return resp

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--label", default="fixer")
    args = ap.parse_args()

    ckpt = os.path.realpath(args.checkpoint)
    state = {"restorer": None, "error": None, "checkpoint": ckpt, "label": args.label}
    try:
        state["restorer"] = Restorer(ckpt, args.src_dir)
        print("[fixer] ready label=%s checkpoint=%s empty_cache=%s"
              % (args.label, ckpt, "on" if EMPTY_CACHE else "off"), flush=True)
    except Exception as exc:                                        # noqa: BLE001
        state["error"] = "%s: %s" % (type(exc).__name__, exc)
        traceback.print_exc()

    build_app(state).run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
