#!/usr/bin/env bash
# Build DrivoR's two metric caches.
#
#   bash cache_metric_drivor.sh           # both: train, then eval
#   bash cache_metric_drivor.sh train     # navtrain -> exp/train_metric_cache   needed by training
#   bash cache_metric_drivor.sh eval      # navtest  -> exp/metric_cache         needed by eval_drivor.sh
#
# Environment: WORKERS (default 32), PROCESS_POOL=false (thread pool instead of processes).
#
# Two different pickles from two different entrypoints; neither can stand in for the other:
#   train  run_train_metric_caching.py (train_cache_processor.py): the per-token PDM assets that
#          score_module/compute_navsim_score.py rolls every proposal against -- the scorer head's
#          GT, for the train and val logs of navtrain. drivor_agent.py reads it from the fixed path
#          $NAVSIM_EXP_ROOT/train_metric_cache, so that is where it goes.
#   eval   run_metric_caching.py (metric_cache_processor.py): what run_pdm_score.py scores navtest
#          against. Its yaml's default cache_path is also train_metric_cache, so the path is always
#          passed explicitly here -- a default run would write into the training GT's directory.
# Both hold ground truth about the scenario only, so one of each serves both arms.
# Neither is the dataset cache (cache_drivor.sh).
#
# Both resume: a token whose metric_cache.pkl exists is skipped (force_feature_computation=false).
# metadata/*.csv, the index the loaders read, is written only after the last token, so a half-built
# cache has none; train_drivor.sh / eval_drivor.sh refuse it, and rerunning this finishes it.
# The caches are gitignored (multi-GB), so a fresh clone must run this before training or eval.
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"

MODE="${1:-all}"
case "$MODE" in
    train|eval|all) ;;
    *) echo "usage: $0 [train|eval|all]" >&2; exit 2 ;;
esac

# Processes, not threads, as in cache_drivor.sh: the metric caching is pure-Python nuplan/shapely
# work that holds the GIL. Set PROCESS_POOL=false to fall back to threads if the pool trips over
# something unpicklable.
WORKERS="${WORKERS:-32}"
PROCESS_POOL="${PROCESS_POOL:-true}"

# build <entrypoint> <split> <cache dir>
build() {
    echo "=== metric cache: $1 split=$2 -> $3 ==="
    "$PY" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/$1" \
        train_test_split=$2 \
        cache.cache_path="$3" \
        worker=single_machine_thread_pool \
        worker.use_process_pool=$PROCESS_POOL \
        worker.max_workers=$WORKERS
    ls "$3"/metadata/*.csv >/dev/null 2>&1 || {
        echo "ERROR: $1 exited without writing $3/metadata/*.csv" >&2; exit 1; }
}

if [ "$MODE" != eval ]; then
    build run_train_metric_caching.py navtrain "$NAVSIM_EXP_ROOT/train_metric_cache"
fi
if [ "$MODE" != train ]; then
    build run_metric_caching.py navtest "$NAVSIM_EXP_ROOT/metric_cache"
fi
