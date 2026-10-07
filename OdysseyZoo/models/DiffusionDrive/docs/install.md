# Installation for DiffusionDrive

After successfully installing the NAVSIM environment, you should further proceed to install the following packages for DiffusionDrive:

```bash
conda activate navsim
pip install diffusers einops 
```

To enable faster download, [Jinkun](https://github.com/Jzzzi) provide a improved script [super_download.sh](../download/super_download.sh) by using tmux to parallelize the download process. Thanks for his contribution!

```bash
cd /path/to/DiffusionDrive/download
bash super_download.sh
```

