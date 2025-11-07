# export TENSORNVME_DEBUG=1

# sudo apt-get install ffmpeg libsm6 libxext6 -y

GPUS_PER_NODE=$ARNOLD_WORKER_GPU
MASTER_ADDR=$METIS_WORKER_0_HOST":"$METIS_WORKER_0_PORT
NNODES=$ARNOLD_WORKER_NUM

CONFIG=${1:-"configs/mimicgen/train/mimicgen_12800.py"}

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank ${ARNOLD_ID:-0} \
    --rdzv_endpoint $MASTER_ADDR \
    --rdzv_backend c10d \
    scripts/train_web.py \
    $CONFIG \
    