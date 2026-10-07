# Installing without the container

This builds the simulator and planner interpreters from the package inventory of the reference
host (`OdysseyBenchmark/configs/deployment/environment-reference.json`) when the frozen image of
[deployment.md](deployment.md) is not available. The Fixer and ReCogDrive interpreters
have their own recipes ([OdysseyRenderer/fixer/README.md](../OdysseyRenderer/fixer/README.md), [installation.md](installation.md) "ReCogDrive").

Verified on 2026-10-02 on a second host: Linux, 5 × NVIDIA B200 (compute capability 10.0),
driver 580.95.05, system CUDA 13.1, no root and no Docker. With these interpreters
`OdysseyBenchmark/tools/check_env.py` passed, all 11 shipped agent configs built (`odyssey_runtime check`), the
debug run completed, and two concurrent runs of `ltf_sdroute` on odyssey_scene009 nr gave
byte-identical images, plans, trajectories and scores. Package versions equal the inventory
except pip and wheel. Another GPU type renders and restores slightly differently from the reference host, so
compare runs from one host ([getting_started.md](getting_started.md) "Reproducibility").

Needs conda (e.g. Miniforge), git, and access to PyPI, download.pytorch.org, github.com and
developer.download.nvidia.com. Run from the checkout:

```bash
export ODYSSEY_RUNTIME_ROOT=/abs/path/odyssey-runtime
R=$ODYSSEY_RUNTIME_ROOT
export TMPDIR=$R/tmp && mkdir -p $TMPDIR      # builds and JIT extensions need an exec-mounted temp dir
```

## 1. CUDA 12.8 toolkit

nvdiffrast and mmcv compile CUDA code at install time and gsplat at first use, against the cu128
torch; a newer system toolkit does not work with it. The runfile installs the toolkit alone,
without root:

```bash
wget -P $TMPDIR https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh $TMPDIR/cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=$R/cuda-12.8 \
  --no-man-page --override --no-drm --no-opengl-libs
export CUDA_HOME=$R/cuda-12.8
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)   # 10.0 on B200
```

Without root the installer reports that it cannot write `/var/log` or the `pkgconfig` files;
the toolkit itself is complete (`$CUDA_HOME/bin/nvcc --version`: 12.8).

## 2. Simulator and planner environments

The pinned lists leave out torch and its CUDA wheels (installed from the cu128 index), and the
packages built below:

```bash
python3 - <<'PY'
import json, os
inv = json.load(open("OdysseyBenchmark/configs/deployment/environment-reference.json"))
skip = {"torch", "torchvision", "triton", "pip", "wheel", "nuplan-devkit", "nvdiffrast", "navsim", "mmcv"}
for env in ("simulator", "planner"):
    pins = [f"{k}=={v}" for k, v in inv[env]["packages"].items()
            if k.lower() not in skip and not k.lower().startswith("nvidia-")]
    open(os.path.join(os.environ["TMPDIR"], f"{env}_pins.txt"), "w").write("\n".join(pins) + "\n")
PY
for env in simulator planner; do
  P=$R/envs/$env/bin/python
  conda create -y -p $R/envs/$env --override-channels -c conda-forge python=3.9.23 pip
  $P -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  $P -m pip install --no-deps -r $TMPDIR/${env}_pins.txt
  $P -m pip install --no-deps "git+https://github.com/motional/nuplan-devkit/@ce3c323af01c0d7ec5672f7832ef53f9c679aab0"
done
```

## 3. CUDA extensions and navsim

```bash
git clone --branch v0.4.0 --depth 1 https://github.com/NVlabs/nvdiffrast third_party/nvdiffrast
PATH=$R/envs/simulator/bin:$CUDA_HOME/bin:$PATH \
  $R/envs/simulator/bin/python -m pip install --no-deps --no-build-isolation ./third_party/nvdiffrast
PATH=$R/envs/planner/bin:$CUDA_HOME/bin:$PATH MMCV_WITH_OPS=1 FORCE_CUDA=1 \
  $R/envs/planner/bin/python -m pip install --no-deps --no-build-isolation mmcv==2.1.0
$R/envs/planner/bin/python -m pip install --no-deps -e OdysseyZoo/models/navsim
```

Both builds target `TORCH_CUDA_ARCH_LIST` only (`cuobjdump --list-elf` on the built `.so` shows
it). gsplat compiles its kernels into `TORCH_EXTENSIONS_DIR` during the first episode, which
then takes a few minutes longer. The planner puts each model's own repository first on its path,
so the installed navsim only fills in for code outside a model repository; the inventory lists
1.1.0, the bundled `OdysseyZoo/models/navsim` is 2.0.0.

## 4. Shell and check

Export the variables of [deployment.md](deployment.md) "Shell" with
`ODYSSEY_SIM_PY=$R/envs/simulator/bin/python`, `ODYSSEY_PLANNER_PY=$R/envs/planner/bin/python`,
`CUDA_HOME=$R/cuda-12.8` and `TORCH_EXTENSIONS_DIR` on an exec-mounted filesystem, then run
`python OdysseyBenchmark/tools/check_env.py --gpus <your GPUs>` and the installation check described there.
