# Understanding the data format and classes

OpenScene is a compact redistribution of the large-scale [nuPlan dataset](https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/index.html), retaining only relevant annotations and sensor data at 2Hz. This reduces the dataset size by a factor of >10. The data used in NAVSIM is structured into `navsim.common.dataclasses.Scene` objects. A `Scene` is a list of `Frame` objects, each containing the required inputs and annotations for training a planning `Agent`.

**Caching.** Evaluating planners involves significant preprocessing of the raw annotation data, including accessing the global map at each ´Frame´ and converting it into a local coordinate system. You can generate the cache with:
```
# upstream script, not included in this release (here: bash cache_metric_drivor.sh eval): https://github.com/valeoai/DrivoR/blob/f02665403df799c1b4ddd8b0d34e073f0555c13a/scripts/evaluation/run_metric_caching.sh
cd $NAVSIM_DEVKIT_ROOT/scripts/
./run_metric_caching.sh
```

This will create the metric cache under `$NAVSIM_EXP_ROOT/metric_cache`, where `$NAVSIM_EXP_ROOT` is defined by the environment variable set during installation.
