#!/bin/bash
# CUDA_VISIBLE_DEVICES=4,5 python -m pipeline.generate --model Llama-2-7b-hf --dataset SQuAD --fraction_of_data_to_use 1 --project_ind for_clustering
# CUDA_VISIBLE_DEVICES=4,5 python -m clustering.nli-deberta-v3-base --model Llama-2-7b-hf --dataset SQuAD --project_ind for_clustering --clustering nli-deberta-v3-base

# usage: bash run.sh [GPU ids]   e.g. bash run.sh 4,5,6,7   (default: 0,1,2,3,4,5,6,7)
set -e
cd "$(dirname "$0")"
trap 'trap - INT TERM; echo "process kill"; kill 0' INT TERM

IFS=',' read -ra GPUS <<< "${1:-4,5,6,7}"
N=${#GPUS[@]}
MODEL=Llama-2-7b-hf
DATASET=SQuAD
OUT=minsuh-output
echo "GPUs: ${GPUS[*]} (N=$N)"

# 1. generate outputs and extract hidden embeddings
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -m pipeline.generate --model $MODEL --dataset $DATASET \
      --project_ind for_clustering_shard$i --num_shards $N --shard_id $i &
done
wait

# 2. merge shards
CUDA_VISIBLE_DEVICES=${GPUS[0]} python - <<EOF
import os, pandas as pd
merged = []
for i in range($N):
    merged += pd.read_pickle('$OUT/${MODEL}_${DATASET}_for_clustering_shard{}/0.pkl'.format(i))
os.makedirs('$OUT/${MODEL}_${DATASET}_for_clustering', exist_ok=True)
pd.to_pickle(merged, '$OUT/${MODEL}_${DATASET}_for_clustering/0.pkl')
print('merged', len(merged))
EOF

# 3. clustering
CUDA_VISIBLE_DEVICES=${GPUS[0]} python -m clustering.nli-deberta-v3-base --model $MODEL --dataset $DATASET \
    --project_ind for_clustering --clustering nli-deberta-v3-base

# 4. AUROC
CUDA_VISIBLE_DEVICES=${GPUS[0]} python func/evalFunc.py
