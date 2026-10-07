# Fixer setup for Odyssey simulation

The launcher restores every rendered camera image with the Fixer preset `fixer_h1b16`: the
fine-tuned `h1_b16_e1_model_11001.pkl` checkpoint run in torch (bf16 autocast). It runs in a
resident Python worker over shared memory, on the same GPU as the simulation. You do not need to
start an HTTP server or open a port.

The Cosmos tokenizer in `models/base/tokenizer_fast.pth` is traced TorchScript whose 81
convolutions were saved with cuDNN benchmark mode on. Benchmark mode times the cuDNN kernels on
the first batch of each process and keeps the fastest, so two processes could restore the same
frame to slightly different images. The worker rewrites those convolutions to benchmark off
before its first request (`cudnn_flags.py`; its log prints `cudnn benchmark disabled on 81
scripted convolutions`). The same input then gives the same image in every process on one
host, also while other processes use the GPU; a restore takes about 10% longer. When the GPU
cannot supply the memory a restore needs, it fails with a CUDA out-of-memory error instead of
choosing other kernels. Another GPU type, driver or Torch build can produce slightly different
images, so compare runs made on one host.

## 1. Checkpoints and interpreter

From the repository root, set (`OdysseyBenchmark/scripts/host_env.example.sh` already sets the
first three):

```bash
export ODYSSEY_FIXER_ROOT="$PWD/OdysseyRenderer/fixer"
export ODYSSEY_RUNTIME_ROOT=/absolute/path/to/odyssey-runtime
export ODYSSEY_FIXER_PY="$ODYSSEY_RUNTIME_ROOT/fixer/bin/python"
export FIXER_MODELS_DIR="$ODYSSEY_FIXER_ROOT/models"
```

Download the weights into this checkout. The fine-tuned checkpoint is published with the planner
weights; the two base files come unchanged from NVIDIA:

```bash
hf download ADRLAB/odyssey-models --include "fixer/*" --local-dir OdysseyRenderer
hf download nvidia/Fixer --include "base/*" --local-dir OdysseyRenderer/fixer/models
(cd OdysseyRenderer/fixer/models && sha256sum -c SHA256SUMS)
```

```text
OdysseyRenderer/fixer/models/
  base/model_fast_tokenizer.pt          from nvidia/Fixer
  base/tokenizer_fast.pth               from nvidia/Fixer
  finetuned/h1_b16_e1_model_11001.pkl   from ADRLAB/odyssey-models (4,054,208,250 bytes)
  NOTICE, LICENSE-NVIDIA-Open-Model-License.txt, SHA256SUMS
```

The fine-tuned checkpoint is a Derivative Model of NVIDIA Fixer under the NVIDIA Open Model
License (Built on NVIDIA Cosmos). It is not the public pretrained Fixer checkpoint and must not be
replaced by it under the same preset. The base files construct the model the fine-tuned weights
are loaded into.

Use the supplied Fixer environment or create a separate Python 3.12 environment:

```bash
conda create -y -p "$ODYSSEY_RUNTIME_ROOT/fixer" python=3.12
"$ODYSSEY_FIXER_PY" -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$ODYSSEY_FIXER_PY" -m pip install --no-deps -r OdysseyRenderer/fixer/requirements_nodeps.txt
"$ODYSSEY_FIXER_PY" -m pip install --no-deps --no-build-isolation OdysseyRenderer/fixer/flash_attn_stub
```

The no-deps install keeps the deployed Torch version instead of letting Cosmos' dependency pins
replace it. `requirements_nodeps.txt` is a deployment snapshot, not a minimal environment.
`shims/` supplies the transformer-engine/megatron interfaces inference uses, in torch. The
flash-attention stub is import-only and raises if executed; this environment is for inference,
not for training.

The worker clears inherited `LD_LIBRARY_PATH` and `PYTHONPATH` and sets its own `shims/` path,
so its libraries must be found through the installed wheels or the system loader.

## 2. Test the worker and a simulation

In a shell that has sourced your environment file ([docs/installation.md](../../docs/installation.md),
"Shell"), with `ODYSSEY_FIXER_GPU=0`. This starts the real
shared-memory worker, restores a three-camera batch, checks shape and type, then closes it:

```bash
export ODYSSEY_FIXER_GPU=0
"$ODYSSEY_SIM_PY" - <<'PY'
import numpy as np
from odyssey_runtime.restorer import SharedRestorer
r = SharedRestorer('fixer_h1b16')
try:
    images = [np.full((576, 1024, 3), 127, np.uint8) for _ in range(3)]
    out = r.restore_batch(images)
    assert len(out) == 3 and all(x.shape == images[0].shape and x.dtype == np.uint8 for x in out)
    print(r.worker.ready, r.last_server_ms)
finally:
    r.close()
PY
"$ODYSSEY_SIM_PY" -m odyssey_runtime run --agent OdysseyBenchmark/agents/ltf_sdroute.yaml --scene odyssey_scene001 --react nr --gpu 0 --max-steps 40
```

The uniform-image test checks transport and loading, not image quality. On failure, read
`fixer_worker_<pid>.log` under the run's `runtime/` directory (for the standalone test above,
in `OdysseyRenderer/fixer/`; delete it afterwards). Common causes are missing base or
checkpoint files, a missing CUDA library, or insufficient GPU memory (the worker uses about 5.5 GB).
