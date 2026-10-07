# OdysseyRenderer

Renders the camera images a planner sees from a reconstructed scene: an OmniRe Gaussian-splatting
scene (static background, ground, rigid and deformable actors, pedestrians, traffic lights) driven
by the simulator's world state, plus the Fixer image restorer that cleans each rendered image.

## Layout

```
odyssey_renderer/        the renderer package
  base_renderer.py       RenderState and the renderer base class the simulator drives
  omnire/                the OmniRe engine (engine.py), its node models, camera and actor
                         contracts, traffic-light control (tl_control.py), restorer client
  mtgs/                  the Gaussian models, rasterisation helpers and checkpoint loading
  ego_lift.py, lidar_ground.py   ego height and road-surface lift
fixer/                   the Fixer restoration worker (fixer_server.py), its model code, shims and
                         setup notes (fixer/README.md); weights go in fixer/models/ (not in git)
tests/                   renderer tests
```

## Entry points

- `odyssey_renderer.omnire.engine.OmniReRenderEngine`: the engine. The simulator builds it
  from its Hydra config (`renderer=omnire`, `_target_` in
  `OdysseyTrafficAgent/odyssey/configs/common/renderer/omnire.yaml`).
- `odyssey_renderer.omnire.restorer_client`: starts or connects to the Fixer worker
  (`fixer/fixer_server.py`, in its own interpreter, `ODYSSEY_FIXER_PY`); `FIXER_PRESETS` names the
  checkpoints under `ODYSSEY_FIXER_ROOT` (default `OdysseyRenderer/fixer`).

## How the others use it

The simulator (OdysseyTrafficAgent) renders through `RenderState` and the engine above; its render
manager also applies the traffic-light control. The benchmark runtime (OdysseyBenchmark) reads
`FIXER_PRESETS` and runs the shared restorer worker. `OdysseyRenderer` must be on `PYTHONPATH`
(the host environment and the launcher set it). The renderer itself imports a few simulator
types (`odyssey.base_class`, `odyssey.scenario`, `odyssey.engine`, `odyssey.utils`), so it runs
inside the simulator's interpreter.

Tests: `python -m pytest -q OdysseyRenderer/tests` from the repository root, in the simulator
interpreter.

## License and provenance

Apache-2.0 (`LICENSE` at the repository root), except: `odyssey_renderer/mtgs/` comes from MTGS
through WorldEngine (Apache-2.0, modified files carry a notice), `omnire/deform_network.py` from
drivestudio (MIT), `fixer/src/` and the restorer in `fixer/fixer_server.py` from NVIDIA Fixer
(Apache-2.0), and `fixer/shims/transformer_engine/` from TransformerEngine (Apache-2.0). The Fixer
weights are under the NVIDIA Open Model License. Details: `THIRD_PARTY_NOTICES.md`.
