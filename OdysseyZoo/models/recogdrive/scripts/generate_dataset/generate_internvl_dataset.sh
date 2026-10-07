set -x

TRAIN_TEST_SPLIT=navtrain

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_dataset.py \
    agent=recogdrive_baseline_agent \
    experiment_name=generate_dataset \
    train_test_split=$TRAIN_TEST_SPLIT 