# Third-party notices

The code in this repository is licensed under the Apache License 2.0 (`LICENSE`), except where
a file or directory below comes from another project; those parts keep their original licence.
Model weights and scenes are distributed separately, with their own licences (see their
Hugging Face model and dataset cards).

## Code in this repository

| Path | Origin | Licence |
|---|---|---|
| `OdysseyTrafficAgent/odyssey/` (the simulator, started from WorldEngine's SimEngine) | [WorldEngine](https://github.com/OpenDriveLab/WorldEngine), modified; each modified file carries a notice | Apache-2.0 |
| `OdysseyTrafficAgent/odyssey/` (scenario description, engine and manager structure) | [MetaDrive](https://github.com/metadriverse/metadrive) | Apache-2.0 |
| `OdysseyTrafficAgent/odyssey/components/agents/policy/pdm_planner/` | PDM-Closed from [tuPlan Garage](https://github.com/autonomousvision/tuplan_garage), Copyright 2023 Daniel Dauner, Marcel Hallgarten, Andreas Geiger and Kashyap Chitta | Apache-2.0 |
| `OdysseyTrafficAgent/odyssey/components/agents/controller/tracker/` (LQR tracker), `pdm_planner/simulation/` (batch LQR, kinematic bicycle) and other nuPlan-based utilities | [nuplan-devkit](https://github.com/motional/nuplan-devkit), Copyright 2021 Motional | Apache-2.0 |
| `OdysseyRenderer/odyssey_renderer/mtgs/` | [MTGS](https://github.com/OpenDriveLab/MTGS) (through WorldEngine), modified; each modified file carries a notice | Apache-2.0 |
| `OdysseyRenderer/odyssey_renderer/omnire/deform_network.py` | [drivestudio](https://github.com/ziyc/drivestudio) `models/modules.py` | MIT (text below) |
| `OdysseyRenderer/fixer/src/`, the restorer in `OdysseyRenderer/fixer/fixer_server.py` | [NVIDIA Fixer](https://github.com/nv-tlabs/Fixer), Copyright 2025 NVIDIA Corporation | Apache-2.0 |
| `OdysseyRenderer/fixer/shims/transformer_engine/` | [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) v2.5, ported to pure PyTorch | Apache-2.0 |
| `OdysseyZoo/models/navsim`, `DrivoR` (with its vendored `nuplan-devkit`), `recogdrive` | [NAVSIM](https://github.com/autonomousvision/navsim), [DrivoR](https://github.com/valeoai/DrivoR), [ReCogDrive](https://github.com/xiaomi-research/recogdrive), modified as stated in `OdysseyZoo/NOTICE` | Apache-2.0 (`LICENSE` in each directory) |
| `OdysseyZoo/models/DiffusionDrive`, `SafeDrive` | [DiffusionDrive](https://github.com/hustvl/DiffusionDrive), SafeDrive, modified as stated in `OdysseyZoo/NOTICE`; SafeDrive also carries `OdysseyBenchmark/patches/safedrive-odysseyzoo-v2.patch` | MIT (`LICENSE` in each directory) |
| `OdysseyZoo/sdroute/` and the rest of `OdysseyZoo/` | OdysseyZoo, Copyright (c) 2026 Seunghoon Yu (`OdysseyZoo/NOTICE`); `sdroute/assets/` holds OpenStreetMap data | see `OdysseyZoo/NOTICE`; OSM data ODbL (below) |

## Installed separately (not in this repository)

| Component | Use | Licence |
|---|---|---|
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) (`third_party/nvdiffrast`) | sky cubemap lookup in the renderer | NVIDIA Source Code License (non-commercial use) |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Gaussian rasterisation | Apache-2.0 |
| [nuplan-devkit](https://github.com/motional/nuplan-devkit) | maps, scenario and simulation types | Apache-2.0; nuPlan data under the [nuPlan terms of use](https://www.nuscenes.org/terms-of-use) |
| [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) | the Fixer model code | Apache-2.0 |
| Fixer weights ([nvidia/Fixer](https://huggingface.co/nvidia/Fixer) base files, the fine-tuned checkpoint in [ADRLAB/odyssey-models](https://huggingface.co/ADRLAB/odyssey-models)) | image restoration | NVIDIA Open Model License; Built on NVIDIA Cosmos |

## Data in this repository

- `OdysseyZoo/sdroute/assets/` contains OpenStreetMap data,
  © OpenStreetMap contributors, available under the
  [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
  ([copyright and licence](https://www.openstreetmap.org/copyright)).
- The traffic-light timetables (`OdysseyBenchmark/data/tlc_timetable/`) and the
  SD-route edges (`OdysseyBenchmark/data/sd_roadblock_map/`) are derived from nuPlan; the nuPlan terms of use apply.

## MIT licence of drivestudio

```
MIT License

Copyright (c) 2024 Ziyu Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
