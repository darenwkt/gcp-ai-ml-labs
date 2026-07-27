#!/bin/bash
set -euo pipefail

# Ensure conda/python binaries are in PATH
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# Parse CLUSTER_SPEC if present to export MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK
if [ -n "${CLUSTER_SPEC:-}" ]; then
    echo "CLUSTER_SPEC is set. Parsing distributed training variables..."
    eval $(python -c "
import os, json
try:
    cluster = json.loads(os.environ.get('CLUSTER_SPEC', '{}')).get('cluster', {})
    task = json.loads(os.environ.get('CLUSTER_SPEC', '{}')).get('task', {})
    pools = sorted(cluster.keys())
    addr, port = cluster.get('workerpool0', ['localhost:29500'])[0].split(':') if 'workerpool0' in cluster else ('localhost', '29500')
    rank, total = 0, 0
    for p in pools:
        if p == task.get('type'):
            rank = total + task.get('index', 0)
        total += len(cluster[p])
    total = max(1, total)
    print(f'export MASTER_ADDR={addr} MASTER_PORT={port} WORLD_SIZE={total} RANK={rank}')
except Exception as e:
    print('# Error parsing CLUSTER_SPEC:', e)
")
else
    echo "CLUSTER_SPEC is not set. Defaulting to single node."
    export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
    export MASTER_PORT=${MASTER_PORT:-"29500"}
    export WORLD_SIZE=${WORLD_SIZE:-1}
    export RANK=${RANK:-0}
fi

# Isolate Triton compiler cache directories to avoid rank locks
export TRITON_CACHE_DIR="/tmp/triton_cache_rank_${RANK}"

echo "Distributed config:"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  RANK=$RANK"

# Now run torchrun
exec torchrun \
    --nnodes="$WORLD_SIZE" \
    --node_rank="$RANK" \
    --rdzv_id="gpt2_training_run" \
    --rdzv_backend="c10d" \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_conf="timeout=900" \
    "$@"
