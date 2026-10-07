import json
from types import SimpleNamespace

import numpy as np
import pytest

from odyssey_benchmark.tlc_filter import load_red_renderable_flags


def test_test_json_overrides_scene_flags(tmp_path):
    path = tmp_path / "renderer_flags.json"
    path.write_text(json.dumps({"scene-a": {"42": [True, False]}}), encoding="utf-8")
    scene = {"id": "scene-a", "traffic_light_red_renderable": {"42": [True, True]}}
    assert load_red_renderable_flags(scene, str(path))["42"] == [True, False]
    with pytest.raises(ValueError):
        load_red_renderable_flags({"id": "scene-a", "traffic_light_red_renderable": {"42": [0]}})
