
#!/bin/bash
set -x

NEW_NUM_NODES=4

NUM_GPUS_PER_NODE=8
RAY_PORT=10000
MASTER_ADDR="xx.xx.xx.xx"

echo "[Head Node] Starting Ray head at $MASTER_ADDR:$RAY_PORT ..."
ray start \
    --head \
    --node-ip-address="$MASTER_ADDR" \
    --port=$RAY_PORT \
    --num-gpus=$NUM_GPUS_PER_NODE \
    --disable-usage-stats \
    --num-cpus=88
