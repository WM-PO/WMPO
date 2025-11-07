#!/bin/bash
set -x

NUM_GPUS_PER_NODE=8
RAY_PORT=10000
MASTER_ADDR="xx.xx.xx.xx"

echo "[Worker Node] Joining Ray cluster at $MASTER_ADDR:$RAY_PORT ..."
ray start \
    --address="$MASTER_ADDR:$RAY_PORT" \
    --num-gpus=$NUM_GPUS_PER_NODE \
    --num-cpus=88