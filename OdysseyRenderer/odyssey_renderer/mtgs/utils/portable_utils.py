# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
"""Helpers for reading scene checkpoints.

``alias_numpy2_pickle_modules`` lets a numpy 1.x interpreter unpickle arrays saved under numpy 2.

``AttrDict`` is a dict whose keys are also attributes. A key with a dot is split at its first dot
into a nested level, so a torch state_dict entry ``"means.weight"`` is read as
``asset.means.weight``. Dicts nested in values, lists and tuples are converted to the same class
(lists and tuples become lists). Each key is stored both as a dict item and as an instance
attribute. ``convert_to_attribute_dict`` wraps a checkpoint dict in one.
"""
import logging
import sys

logger = logging.getLogger(__name__)


def alias_numpy2_pickle_modules():
    """Let a numpy-1.x interpreter unpickle arrays written by numpy >= 2.0.

    numpy 2 renamed numpy.core to numpy._core, and a pickled array names its reconstructor by
    module path. Some scene checkpoints were saved under numpy 2, so a numpy 1.x env fails in
    torch.load with `No module named 'numpy._core'`. Both names refer to the same
    reconstructors, so aliasing changes no decoded value. A no-op on numpy >= 2.
    """
    try:
        import numpy._core.multiarray  # noqa: F401
        return
    except ImportError:
        pass

    import numpy.core

    sys.modules.setdefault("numpy._core", numpy.core)
    for sub in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        mod = getattr(numpy.core, sub, None)
        if mod is not None:
            sys.modules.setdefault("numpy._core." + sub, mod)


# Public methods of the class itself; every other non-dunder class attribute is copied onto the
# instance when it is built.
_METHODS = ("update", "pop")


def _convert(cls, value):
    if isinstance(value, (list, tuple)):
        return [cls(item) if isinstance(item, dict) else item for item in value]
    if isinstance(value, dict) and not isinstance(value, cls):
        return cls(value)
    return value


def _merge(target, key, value):
    """Fold a plain ``key`` into the level already built from dotted ``key.*`` entries."""
    assert isinstance(value, dict), f"Conflicting value for key `{key}` found. Value should be a dict."
    for subkey in value:
        if subkey in target:
            assert target[subkey] == value[subkey], f"Conflicting value for key `{key}.{subkey}` found."
            logger.warning(f"Duplicate key `{key}.{subkey} found.")
    target.update(value)


def _nest_dotted_keys(source):
    levels = {}
    for key, value in source.items():
        if "." in key:
            head, rest = key.split(".", 1)
            if head not in levels:
                levels[head] = {}
            levels[head][rest] = value
        elif key in levels:
            _merge(levels[key], key, value)
        else:
            levels[key] = value
    return levels


class AttrDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for key, value in _nest_dotted_keys(d).items():
            setattr(self, key, value)
        for name in self.__class__.__dict__.keys():
            if not (name.startswith("__") and name.endswith("__")) and name not in _METHODS:
                setattr(self, name, getattr(self, name))

    def __setattr__(self, name, value):
        value = _convert(self.__class__, value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        items = e or dict()
        items.update(f)
        for key in items:
            setattr(self, key, items[key])

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super().pop(k, d)


def convert_to_attribute_dict(obj: dict) -> AttrDict:
    return AttrDict(obj)
