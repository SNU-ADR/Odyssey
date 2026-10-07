"""Renderer red-light flags for the TLC metric (tlc_metric.score_tlc's render waiver).

Flags are indexed by scene frame.  A flag of False means that the renderer could not show RED
for that connector at that frame; True means it could.  Missing flags are unknown and never
waive a violation.  No published scene carries these flags, so the waiver currently has no effect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


def load_red_renderable_flags(scene: Mapping, override_path: Optional[str] = None) -> dict:
    """Read a per-connector flag array from the scene or an optional test JSON file.

    A scene may carry ``scene['traffic_light_red_renderable']``; no published scene does.
    The test JSON has the shape ``{scene_id: {connector_id: [true, false, null]}}``.
    The override replaces the scene payload for that scene, not individual connectors.
    """
    flags = scene.get("traffic_light_red_renderable", {})
    if override_path:
        with Path(override_path).open(encoding="utf-8") as file:
            scenes = json.load(file)
        if not isinstance(scenes, dict):
            raise ValueError("TLC render sanity JSON must map scene IDs to connector arrays")
        flags = scenes.get(str(scene["id"]), flags)
    if not isinstance(flags, dict):
        raise ValueError("traffic_light_red_renderable must map connector IDs to arrays")
    normalized = {}
    for connector_id, values in flags.items():
        if not isinstance(values, (list, tuple, np.ndarray)) or any(
            value is not None and not isinstance(value, (bool, np.bool_)) for value in values
        ):
            raise ValueError(
                f"render sanity for connector {connector_id} must be an array of booleans/nulls"
            )
        normalized[str(connector_id)] = [None if value is None else bool(value) for value in values]
    return normalized
