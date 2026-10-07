import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[3]

def installer():
    path = ROOT / "OdysseyBenchmark/scripts/apply_safedrive_patch.py"
    assert path.is_file(), "checked native patch installer missing"
    spec = importlib.util.spec_from_file_location("native_patch_installer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_refuses_unknown_source_without_modification(tmp_path):
    mod = installer()
    folder = tmp_path / "navsim/agents/safedrive"
    folder.mkdir(parents=True)
    p = folder / "safedrive_model.py"
    p.write_text("unexpected upstream revision")
    with pytest.raises(ValueError, match="source"):
        mod.apply(tmp_path)
    assert p.read_text() == "unexpected upstream revision"
    assert not (tmp_path / ".odyssey-safedrive-patch.json").exists()

def test_pinned_patch_install_is_checked_and_idempotent(tmp_path):
    mod = installer()
    manifest = json.loads(mod.MANIFEST.read_text())
    source = ROOT / "OdysseyZoo/models/SafeDrive"
    if not source.is_dir():
        pytest.skip("native SafeDrive source not installed")
    for name, identity in manifest["files"].items():
        path = source / name
        if path.exists():
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    # A deployed source may already be patched; reconstruct the pinned base.
    first = next(iter(manifest["files"]))
    if hashlib.sha256((tmp_path / first).read_bytes()).hexdigest() == manifest["files"][first]["after"]:
        subprocess.run(["git", "apply", "-R", str(mod.MANIFEST.with_suffix(".patch"))],
                       cwd=tmp_path, check=True)
    mod.apply(tmp_path)
    snapshots = {name: (tmp_path / name).read_bytes() for name in manifest["files"]}
    mod.apply(tmp_path)
    assert snapshots == {name: (tmp_path / name).read_bytes() for name in manifest["files"]}
    for name, identity in manifest["files"].items():
        assert hashlib.sha256(snapshots[name]).hexdigest() == identity["after"]
    assert (tmp_path / ".odyssey-safedrive-patch.json").is_file()
    helper = ROOT / "OdysseyBenchmark/odyssey_runtime/safedrive_compat.py"
    assert helper.read_bytes() == (tmp_path / "navsim/agents/safedrive/inference_compat.py").read_bytes()
