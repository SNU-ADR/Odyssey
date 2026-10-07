import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
def checker():
    path = ROOT / "OdysseyBenchmark/scripts/check_deployment.py"
    assert path.is_file(), "deployment checker missing"
    spec = importlib.util.spec_from_file_location("deployment_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_missing_or_wrong_checkpoint_is_reported(tmp_path):
    mod = checker()
    record = {"path": "weights.bin", "bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    manifest = {"schema": 1, "files": [record]}
    assert mod.check(manifest, tmp_path, hashes=True)
    (tmp_path / "weights.bin").write_bytes(b"evil")
    assert mod.check(manifest, tmp_path, hashes=True)
    (tmp_path / "weights.bin").write_bytes(b"good")
    assert mod.check(manifest, tmp_path, hashes=True) == []

def test_fast_check_still_rejects_size_mismatch(tmp_path):
    mod = checker()
    (tmp_path / "weights.bin").write_bytes(b"short")
    manifest = {"schema": 1, "files": [{"path": "weights.bin", "bytes": 9, "sha256": "unused"}]}
    assert mod.check(manifest, tmp_path, hashes=False)

def test_source_tree_change_is_detected(tmp_path):
    mod = checker()
    tree = tmp_path / "model"
    tree.mkdir()
    source = tree / "model.py"
    source.write_text("value = 1\n")
    manifest = {"schema": 1, "files": [], "source_trees": [{"path": "model", "sha256": mod.tree_digest(tree)}]}
    assert mod.check(manifest, tmp_path, hashes=True) == []
    source.write_text("value = 2\n")
    assert mod.check(manifest, tmp_path, hashes=True)

def test_an_optional_source_tree_is_checked_only_where_it_exists(tmp_path):
    mod = checker()
    manifest = {"schema": 1, "files": [], "source_trees": [{"path": "nvdiffrast", "sha256": "x", "optional": True}]}
    assert mod.check(manifest, tmp_path, hashes=True) == []
    (tmp_path / "nvdiffrast").mkdir()
    (tmp_path / "nvdiffrast/setup.py").write_text("changed = True\n")
    assert mod.check(manifest, tmp_path, hashes=True) == ["source digest differs: nvdiffrast"]

def test_downloads_and_outputs_beside_a_model_do_not_change_its_source_digest(tmp_path):
    mod = checker()
    tree = tmp_path / "model"
    (tree / "nuplan/weights").mkdir(parents=True)
    (tree / "model.py").write_text("value = 1\n")
    (tree / "nuplan/weights/layer.py").write_text("source = True\n")     # a source dir named weights
    before = mod.tree_digest(tree)
    for extra in ("weights/dinov2/config.json", "ckpts/x.yaml", "exp/run/config.yaml",
                  "cache_train/meta.json", "lightning_logs/version_0/hparams.yaml",
                  "build/lib/model.py", "model.egg-info/x.json"):
        (tree / extra).parent.mkdir(parents=True, exist_ok=True)
        (tree / extra).write_text("{}\n")
    assert mod.tree_digest(tree) == before
    (tree / "nuplan/weights/layer.py").write_text("source = False\n")
    assert mod.tree_digest(tree) != before

def test_scene_and_map_entries_resolve_against_their_roots(tmp_path):
    mod = checker()
    scenes, maps = tmp_path / "scenes", tmp_path / "maps"
    (scenes / "odyssey_scene001").mkdir(parents=True)
    maps.mkdir()
    (scenes / "odyssey_scene001/route.npz").write_bytes(b"route")
    (maps / "nuplan-maps-v1.0.json").write_bytes(b"{}")
    manifest = {"schema": 1, "files": [
        {"base": "scenes", "path": "odyssey_scene001/route.npz", "bytes": 5, "sha256": hashlib.sha256(b"route").hexdigest()},
        {"base": "maps", "path": "nuplan-maps-v1.0.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}]}
    assert mod.check(manifest, tmp_path, hashes=True, scenes_root=scenes, maps_root=maps) == []
    errors = mod.check(manifest, tmp_path, hashes=True)
    assert any("ODYSSEY_SCENES_ROOT" in e for e in errors) and any("NUPLAN_MAPS_ROOT" in e for e in errors)

def test_published_weights_resolve_against_models_root(tmp_path):
    mod = checker()
    models = tmp_path / "models"
    (models / "models/navsim/ckpts").mkdir(parents=True)
    (models / "models/navsim/ckpts/ltf_sdroute.ckpt").write_bytes(b"good")
    record = {"base": "models", "path": "models/navsim/ckpts/ltf_sdroute.ckpt",
              "bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    manifest = {"schema": 1, "files": [record]}
    assert mod.check(manifest, tmp_path, hashes=True, models_root=models) == []
    # Without a models root the published layout is looked up inside ROOT/OdysseyZoo.
    assert mod.check(manifest, tmp_path, hashes=True)

def test_cached_backbone_resolves_against_hf_cache(tmp_path):
    mod = checker()
    cache = tmp_path / "hub"
    (cache / "models--timm--x/snapshots/0").mkdir(parents=True)
    (cache / "models--timm--x/snapshots/0/model.safetensors").write_bytes(b"good")
    record = {"base": "hf_cache", "path": "models--timm--x/snapshots/0/model.safetensors",
              "bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    manifest = {"schema": 1, "files": [record]}
    assert mod.check(manifest, tmp_path, hashes=True, hf_cache=cache) == []
    assert mod.check(manifest, tmp_path, hashes=True, hf_cache=tmp_path / "empty")
