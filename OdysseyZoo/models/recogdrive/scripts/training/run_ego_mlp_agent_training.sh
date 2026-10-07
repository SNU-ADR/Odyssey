export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"

TRAIN_TEST_SPLIT=navtrain

CUDA_VISIBLE_DEVICES=1  python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=ego_status_mlp_agent \
    experiment_name=training_ego_mlp_agent \
    trainer.params.max_epochs=50 \
    train_test_split=navtrain \
    cache_path="" \
    use_cache_without_dataset=True \
    force_cache_computation=False