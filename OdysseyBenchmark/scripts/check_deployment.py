"""Verify explicit external deployment inputs without loading a model or GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path

SOURCE_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".cpp", ".cu", ".h", ".hpp", ".cuh"}

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

#: Not part of a source tree: VCS data, bytecode and package metadata anywhere, and what a
#: directory gathers beside its code once installed and used: a pip build (nvdiffrast's
#: build/), and in OdysseyZoo (its .gitignore) downloaded weights, checkpoints, experiment
#: output and caches. The DINOv2 download into DrivoR/weights, for example, must not change
#: the digest of the DrivoR source.
SKIP_ANYWHERE = {".git", "__pycache__", "lightning_logs"}
SKIP_TOP = {"build", "ckpts", "weights", "pretrained", "exp", "exp_smoke", "_smoke", ".cache"}


def source_file(path, root):
    parts = path.relative_to(root).parts
    return (path.suffix in SOURCE_EXTENSIONS and not SKIP_ANYWHERE.intersection(parts)
            and not any(part.endswith(".egg-info") for part in parts)
            and parts[0] not in SKIP_TOP and not parts[0].startswith("cache_"))


def tree_digest(root):
    value = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and source_file(p, root))
    for path in files:
        value.update(path.relative_to(root).as_posix().encode() + b"\0")
        value.update(digest(path).encode() + b"\n")
    return value.hexdigest()

def check(manifest, root, hashes=False, models_root=None, hf_cache=None, scenes_root=None, maps_root=None):
    if manifest.get("schema") != 1:
        raise ValueError("unsupported deployment manifest schema")
    root = Path(root)
    # "base" names where an entry lives outside the checkout:
    #   models    the published planner weights (ADRLAB/odyssey-models layout)
    #   hf_cache  the Hugging Face hub cache (HF_HUB_CACHE)
    #   scenes    the published scenes (ADRLAB/odyssey-scenes, ODYSSEY_SCENES_ROOT)
    #   maps      one nuPlan map directory (NUPLAN_MAPS_ROOT)
    bases = {"models": Path(models_root) if models_root else root / "OdysseyZoo",
             "hf_cache": Path(hf_cache) if hf_cache else Path.home() / ".cache/huggingface/hub",
             "scenes": Path(scenes_root) if scenes_root else None,
             "maps": Path(maps_root) if maps_root else None}
    unset = {"scenes": "the scenes root is not set (ODYSSEY_SCENES_ROOT or --scenes-root)",
             "maps": "the map directory is not set (NUPLAN_MAPS_ROOT or --maps-root)"}
    errors = []
    for item in manifest["files"]:
        base = item.get("base")
        if base in bases and bases[base] is None:
            if unset[base] not in errors:
                errors.append(unset[base])
            continue
        path = bases.get(base, root) / item["path"]
        name = str(path) if item.get("base") else item["path"]
        if not path.is_file():
            errors.append(f"missing file: {name}")
        elif path.stat().st_size != item["bytes"]:
            errors.append(f"size differs: {name}")
        elif hashes and digest(path) != item["sha256"]:
            errors.append(f"SHA256 differs: {name}")
    for item in manifest.get("source_trees", []):
        path = root / item["path"]
        if not path.is_dir():
            # An optional tree (nvdiffrast's checkout) is checked only where it exists: a container
            # can carry the installed package instead.
            if not item.get("optional"):
                errors.append(f"missing source: {item['path']}")
        elif hashes and tree_digest(path) != item["sha256"]:
            errors.append(f"source digest differs: {item['path']}")
    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "configs/deployment/odysseyzoo-reference.json")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--models-root", type=Path, default=os.environ.get("ODYSSEY_MODELS_ROOT"),
                        help="Published planner weights root (default: ODYSSEY_MODELS_ROOT, else ROOT/OdysseyZoo)")
    parser.add_argument("--hf-cache", type=Path, default=os.environ.get("HF_HUB_CACHE"),
                        help="Hugging Face hub cache (default: HF_HUB_CACHE, else ~/.cache/huggingface/hub)")
    parser.add_argument("--scenes-root", type=Path, default=os.environ.get("ODYSSEY_SCENES_ROOT"),
                        help="Published scenes (default: ODYSSEY_SCENES_ROOT)")
    parser.add_argument("--maps-root", type=Path, default=os.environ.get("NUPLAN_MAPS_ROOT"),
                        help="One nuPlan map directory (default: NUPLAN_MAPS_ROOT)")
    parser.add_argument("--hashes", action="store_true", help="Read all checkpoint bytes and verify source trees")
    args = parser.parse_args()
    errors = check(json.loads(args.manifest.read_text()), args.root, args.hashes, args.models_root, args.hf_cache,
                   args.scenes_root, args.maps_root)
    print(json.dumps({"ok": not errors, "hashes_checked": args.hashes, "errors": errors}, indent=2))
    raise SystemExit(bool(errors))
