"""An agent config: one YAML file that ports a model.

The file carries a ``model`` block (where the model lives and how to start it) and the model's
input/output contract (the same fields as the runtime profile). The loader turns it into the
effective profile JSON the simulator and planner worker already consume, and into the
environment that selects the model. Example::

    model:
      name: ltf_sdroute                      # a label: job names, ODYSSEY_PLANNER, launch.json
      python: ${ODYSSEY_PLANNER_PY}          # the planner interpreter for this model
      repo: ${ODYSSEY_ZOO_ROOT}/models/navsim
      agent_config: ltf_sdroute_agent        # <repo>/navsim/planning/script/config/common/agent/<name>.yaml
      checkpoint: ${ODYSSEY_ZOO_ROOT}/models/navsim/ckpts/ltf_sdroute.ckpt
      overrides: []                          # native Hydra overrides (not env-expanded)
      # adapter: my_adapter.py:MyPlanner     # optional; a NavsimPlanner subclass by file or module
      # extra_pythonpath: [../side_packages] # optional; planner process only
    history_times_s: [-1.5, -1.0, -0.5, 0.0]
    camera_times_s: {CAM_F0: [0.0], CAM_L0: [0.0], CAM_R0: [0.0]}
    navigation: {driving_command: false, sd_route: targets}
    output: {plan_dt: 0.5, shape: [8, 3]}
    feature_shapes: {camera_feature: [1, 3, 256, 1024], status_feature: [1, 8]}   # optional

Relative paths resolve against the file's directory; ``${VAR}``, ``$VAR`` and ``~`` expand in
path fields and ``python``. Fields fixed by the benchmark (world cadence, native rendering and
preprocessing, ego-future output) are not configurable and are filled in here.
"""
from dataclasses import dataclass, replace
import os
import shutil
from pathlib import Path
import re

import yaml

from .profile import ModelProfile

# The benchmark's fixed contract; a config cannot change these.
HIDDEN = dict(version=1, sim_dt=0.1, planning_interval_s=0.1, render_resolution="native",
              preprocessing="model_native", shared_capacity_mb=64, record_images=False,
              record_legacy_ipc=False, audit=False)
TOP_KEYS = {"model", "history_times_s", "camera_times_s", "navigation", "output", "feature_shapes",
            "seed", "stateful", "reset_method"}
MODEL_KEYS = {"name", "python", "repo", "agent_config", "checkpoint", "overrides", "adapter",
              "extra_pythonpath"}
REQUIRED_MODEL_KEYS = {"name", "python", "repo", "agent_config", "checkpoint"}
AGENT_YAML_DIR = "navsim/planning/script/config/common/agent"
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def expand(value, field, environ):
    """Expand ``${VAR}``/``$VAR`` and ``~``; an unset variable is an error naming the field."""
    def sub(match):
        name = match.group(1) or match.group(2)
        if name not in environ:
            raise ValueError(f"model.{field}: environment variable {name} is not set")
        return environ[name]

    return os.path.expanduser(_VAR.sub(sub, str(value)))


def resolve_path(value, base_dir, field, environ):
    """Expand, then resolve relative to the config's directory. abspath keeps symlink spelling."""
    text = expand(value, field, environ)
    if not text:
        raise ValueError(f"model.{field} must not be empty")
    return os.path.abspath(os.path.join(base_dir, text))


def resolve_interpreter(value, base_dir, environ):
    """Like resolve_path for a path; a bare command name (no slash) is looked up on PATH."""
    text = expand(value, "python", environ)
    if not text:
        raise ValueError("model.python must not be empty")
    if os.sep not in text:
        return shutil.which(text, path=environ.get("PATH")) or text
    return os.path.abspath(os.path.join(base_dir, text))


def parse_adapter(value, base_dir, environ):
    """'' (generic), 'file.py:Class' resolved against the config dir, or 'pkg.module:Class'."""
    if value in (None, ""):
        return ""
    spec = expand(value, "adapter", environ)
    target, sep, name = spec.rpartition(":")
    if not sep or not target or not name or not name.isidentifier():
        raise ValueError(f"model.adapter must be 'file.py:Class' or 'package.module:Class', got {value!r}")
    if target.endswith(".py") or os.sep in target:
        target = os.path.abspath(os.path.join(base_dir, target))
    return f"{target}:{name}"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    python: str
    repo: str
    agent_config: str
    checkpoint: str
    overrides: tuple = ()
    adapter: str = ""
    extra_pythonpath: tuple = ()

    @property
    def agent_yaml(self):
        return os.path.join(self.repo, AGENT_YAML_DIR, self.agent_config + ".yaml")


@dataclass(frozen=True)
class AgentConfig:
    path: Path
    model: ModelSpec
    profile: ModelProfile

    def with_model(self, **changes):
        """Deployment overrides from the command line (--repo/--checkpoint/--agent-config)."""
        changes = {k: v for k, v in changes.items() if v is not None}
        if not changes:
            return self
        model = replace(self.model, **{k: (os.path.abspath(v) if k in ("repo", "checkpoint") else v)
                                       for k, v in changes.items()})
        return replace(self, model=model)

    def with_seed(self, seed):
        if seed is None:
            return self
        return replace(self, profile=ModelProfile(dict(self.profile.data, seed=seed)))

    def with_profile(self, **changes):
        return replace(self, profile=ModelProfile(dict(self.profile.data, **changes)))

    def verify_paths(self):
        """Problems that would only surface once the simulator has spent minutes starting."""
        m, problems = self.model, []
        if not (os.path.isfile(m.python) and os.access(m.python, os.X_OK)):
            problems.append(f"model.python is not an executable file: {m.python}")
        if not os.path.isdir(m.repo):
            problems.append(f"model.repo is not a directory: {m.repo}")
        elif not os.path.isfile(m.agent_yaml):
            problems.append(f"model.agent_config {m.agent_config!r} has no yaml at {m.agent_yaml}")
        if not os.path.isfile(m.checkpoint):
            problems.append(f"model.checkpoint is not a file: {m.checkpoint}")
        adapter_file = m.adapter.rpartition(":")[0]
        if adapter_file and (adapter_file.endswith(".py") or os.sep in adapter_file) \
                and not os.path.isfile(adapter_file):
            problems.append(f"model.adapter file not found: {adapter_file}")
        for p in m.extra_pythonpath:
            if not os.path.isdir(p):
                problems.append(f"model.extra_pythonpath entry is not a directory: {p}")
        return problems

    def planner_env(self):
        """Environment that selects this model for a run."""
        m = self.model
        env = dict(ODYSSEY_PLANNER=m.name, ODYSSEY_PLANNER_PY=m.python, ODYSSEY_PLANNER_REPO=m.repo,
                   ODYSSEY_PLANNER_CKPT=m.checkpoint, ODYSSEY_PLANNER_CFG=m.agent_config,
                   ODYSSEY_PLANNER_ADAPTER=m.adapter)
        if m.extra_pythonpath:
            env["ODYSSEY_PLANNER_PYTHONPATH"] = os.pathsep.join(m.extra_pythonpath)
        return env


def is_agent_config(path):
    return str(path).endswith((".yaml", ".yml"))


ROOT = Path(__file__).resolve().parents[2]          # the repository root


def load(path, *, environ=None):
    path = Path(path)
    environ = dict(os.environ if environ is None else environ)
    # The launcher's defaults: OdysseyZoo linked in the checkout, weights beside its code.
    environ.setdefault("ODYSSEY_ZOO_ROOT", str(ROOT / "OdysseyZoo"))
    environ.setdefault("ODYSSEY_MODELS_ROOT", environ["ODYSSEY_ZOO_ROOT"])
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: not valid YAML: {error}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the agent config must be a mapping")
    unknown = set(raw) - TOP_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown settings {sorted(unknown)}; known: {sorted(TOP_KEYS)}")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{path}: a 'model' mapping is required")
    unknown = set(model) - MODEL_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown model settings {sorted(unknown)}; known: {sorted(MODEL_KEYS)}")
    missing = REQUIRED_MODEL_KEYS - set(model)
    if missing:
        raise ValueError(f"{path}: model requires {sorted(missing)}")
    base_dir = str(path.resolve().parent)
    name = str(model["name"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"{path}: model.name must be a plain label (letters, digits, _ . -), got {name!r}")
    overrides = model.get("overrides") or []
    if not isinstance(overrides, list) or any(not isinstance(x, str) or not x for x in overrides):
        raise ValueError(f"{path}: model.overrides must be a list of Hydra override strings")
    extra = model.get("extra_pythonpath") or []
    if not isinstance(extra, list):
        raise ValueError(f"{path}: model.extra_pythonpath must be a list of directories")
    try:
        spec = ModelSpec(
            name=name,
            python=resolve_interpreter(model["python"], base_dir, environ),
            repo=resolve_path(model["repo"], base_dir, "repo", environ),
            agent_config=str(model["agent_config"]),
            checkpoint=resolve_path(model["checkpoint"], base_dir, "checkpoint", environ),
            overrides=tuple(overrides),
            adapter=parse_adapter(model.get("adapter"), base_dir, environ),
            extra_pythonpath=tuple(resolve_path(p, base_dir, "extra_pythonpath", environ) for p in extra),
        )
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    output = raw.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"{path}: 'output' must declare plan_dt and shape")
    data = dict(HIDDEN, stateful=False, reset_method=None)
    for key in ("history_times_s", "camera_times_s", "navigation", "feature_shapes", "seed",
                "stateful", "reset_method"):
        if key in raw:
            data[key] = raw[key]
    data["output"] = dict(output, coordinates="ego_future")
    data["planner_id"] = name
    data["model_overrides"] = list(overrides)
    try:
        profile = ModelProfile(data)
    except (KeyError, ValueError, TypeError) as error:
        detail = f"missing {error}" if isinstance(error, KeyError) else error
        raise ValueError(f"{path}: {detail}") from None
    return AgentConfig(path=path, model=spec, profile=profile)
