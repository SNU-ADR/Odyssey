"""Apply the pinned native correction to an explicitly selected SafeDrive copy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]          # OdysseyBenchmark/
MANIFEST = ROOT / "patches/safedrive-odysseyzoo-v2.json"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def apply(repo, manifest_path=MANIFEST):
    repo = Path(repo).resolve()
    manifest_path = Path(manifest_path).resolve()
    patch = manifest_path.with_suffix(".patch")
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    current = {name: digest(repo / name) for name in files}
    expected = {name: identity["after"] for name, identity in files.items()}
    if digest(ROOT / "odyssey_runtime/safedrive_compat.py") != files["navsim/agents/safedrive/inference_compat.py"]["after"]:
        raise ValueError("source helper and pinned patch differ; regenerate and review the patch")
    if current != expected:
        if current != {name: identity["before"] for name, identity in files.items()}:
            raise ValueError("native source differs from the pinned revision; refusing to patch")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=repo, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=repo, check=True)
        if {name: digest(repo / name) for name in files} != expected:
            raise RuntimeError("installed native patch checksum mismatch")
    (repo / ".odyssey-safedrive-patch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(json.dumps(apply(args.repo, manifest_path=args.manifest), indent=2))
