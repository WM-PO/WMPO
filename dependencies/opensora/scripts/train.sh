# export TENSORNVME_DEBUG=1

# export http_proxy=http://bj-rd-proxy.byted.org:3128
# export https_proxy=http://bj-rd-proxy.byted.org:3128

export WANDB_API_KEY=1ffba3f6afe0d59ce6267833cd36a695f3719b25

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
    # --ckpt-path \
    # /mnt/hdfs/zhufangqi/pretrained_models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors \
    # /mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/world-model/Open-Sora/pretrained_models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors
    