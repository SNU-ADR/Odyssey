"""Resolve the planner adapter class the worker instantiates.

An agent config names its adapter in one of three forms; an empty spec means the generic
``NavsimPlanner``, which runs any native navsim ``AbstractAgent`` from the config alone.

    ""                                   the generic adapter
    my_adapter.py:MyPlanner              a file anywhere (its directory joins sys.path, so
                                         sibling modules import as usual)
    pkg.module:MyPlanner                 an importable module (the repo's own adapters use
                                         relative imports, which a file import cannot satisfy)
"""
import hashlib
import importlib
import importlib.util
import os
import sys


def load_adapter(spec=""):
    from odyssey_bridge.planners.base import NavsimPlanner

    spec = str(spec or "").strip()
    if not spec:
        return NavsimPlanner
    target, sep, name = spec.rpartition(":")
    if not sep or not target or not name:
        raise ValueError(
            f"adapter must be 'file.py:Class' or 'package.module:Class', got {spec!r}")
    if target.endswith(".py") or os.sep in target:
        path = os.path.abspath(target)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"adapter file not found: {path}")
        modname = "odyssey_agent_" + hashlib.sha1(path.encode()).hexdigest()[:12]
        module = sys.modules.get(modname)
        if module is None:
            folder = os.path.dirname(path)
            if folder not in sys.path:
                sys.path.insert(0, folder)
            loaded = importlib.util.spec_from_file_location(modname, path)
            module = importlib.util.module_from_spec(loaded)
            sys.modules[modname] = module
            try:
                loaded.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(modname, None)
                raise
    else:
        module = importlib.import_module(target)
    cls = getattr(module, name, None)
    if not (isinstance(cls, type) and issubclass(cls, NavsimPlanner)):
        raise TypeError(f"{spec}: {name!r} is not a NavsimPlanner subclass")
    return cls
