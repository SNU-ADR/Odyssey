#!/usr/bin/env python3
"""Build a training cache root that holds all three files per token, without touching the hidden-state cache.

navsim's CacheOnlyDataset only accepts a token when every builder's file sits in the same token
directory. The hidden-state cache is left read-only, so the merged root holds only symlinks: two
into the hidden-state cache and one into the route cache:

    <out>/<log>/<token>/internvl_feature.gz  -> <hidden>/<log>/<token>/internvl_feature.gz
                        trajectory_target.gz -> <hidden>/<log>/<token>/trajectory_target.gz
                        sdroute_target.gz    -> <route>/<log>/<token>/sdroute_target.gz

Symlinks, not copies: training only reads. Never run dataset caching against this root -- navsim
rewrites target files, and a symlinked target would be written through into the hidden-state cache.
"""
import argparse
import os
from pathlib import Path

HIDDEN_FILES = ("internvl_feature.gz", "trajectory_target.gz")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", required=True, help="existing hidden-state cache root (read-only)")
    ap.add_argument("--route", required=True, help="SD route cache root")
    ap.add_argument("--out", required=True, help="merged root to create")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hidden, route, out = Path(args.hidden), Path(args.route), Path(args.out)
    made = missing_hidden = missing_route = 0
    logs = sorted(p.name for p in route.iterdir() if p.is_dir())
    for log in logs:
        for token_dir in sorted((route / log).iterdir()):
            token = token_dir.name
            src_hidden = hidden / log / token
            if not all((src_hidden / f).exists() for f in HIDDEN_FILES):
                missing_hidden += 1
                continue
            if not (token_dir / "sdroute_target.gz").exists():
                missing_route += 1
                continue
            dst = out / log / token
            if args.dry_run:
                made += 1
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for f in HIDDEN_FILES:
                link = dst / f
                if not link.is_symlink() and not link.exists():
                    os.symlink((src_hidden / f).resolve(), link)
            link = dst / "sdroute_target.gz"
            if not link.is_symlink() and not link.exists():
                os.symlink((token_dir / "sdroute_target.gz").resolve(), link)
            made += 1
    print(f"{'would create' if args.dry_run else 'created'} {made} token dirs; "
          f"skipped {missing_hidden} without hidden-state files, {missing_route} without a route file")


if __name__ == "__main__":
    main()
