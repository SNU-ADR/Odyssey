export MASTER_PORT=12345

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63666}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"

torchrun \
    --master_addr=$IDP_MASTER_ADDR \
    --nnodes=1 \
    --nproc_per_node=8 \
    --max_restarts=2 \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/plt_all_vis.py