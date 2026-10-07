# Installation

SafeDrive builds on the NAVSIM devkit, so install the
[NAVSIM environment](https://github.com/autonomousvision/navsim?tab=readme-ov-file#getting-started-)
first, then add the packages below.

```bash
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .

# torch and mmcv have to agree: mmcv 2.1.0 is built against torch 2.1
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install openmim
mim install mmcv==2.1.0
mim install mmdet==3.3.0          # needs mmcv >=2.0.0rc4,<2.2.0
pip install spconv-cu118==2.3.6   # the wheel's CUDA suffix must match torch's CUDA
pip install einops lmdb
```

A mismatch between the installed torch and the torch mmcv was compiled against
shows up as an `mmcv._ext` ABI error on the first import, not at install time.

`mmdet` (detection loss) and `spconv` (SECOND lidar branch) are imported
unconditionally, so the camera-only configs need them too. `spconv` ships one
wheel per CUDA version (`spconv-cu118`, `spconv-cu120`, ...). The commands above
follow upstream's install; the released `baseline` / `sdroute` checkpoints were
verified in a newer environment: torch 2.7.1+cu128 with mmcv 2.1.0, mmdet 3.3.0
and spconv-cu120 2.3.6.

## Dataset

Download the OpenScene / navsim splits with the scripts in
[`download/`](../download). [`super_download.sh`](../download/super_download.sh)
parallelises the downloads through tmux.

```bash
cd download
bash super_download.sh
```

The scripts at the repository root take the dataset root from `DATA_ROOT`
(default `./dataset`, maps at `$DATA_ROOT/maps`), derive the NAVSIM environment
variables from it, and write to `./exp`:

```bash
DATA_ROOT=/path/to/dataset bash cache_safedrive.sh
```

The `paper` variant also loads LiDAR `.pcd` files through the relative path
`dataset/sensor_blobs/{trainval,test}`, so keep a `dataset` symlink at the
repository root when `DATA_ROOT` points elsewhere.
