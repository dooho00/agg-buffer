#!/bin/bash
# This script runs training and evaluation for Cora dataset 10 times.
gpu_num=4 # Set the GPU number to use, change as needed

cd "$(dirname "$0")/.." 
export PYTHONPATH=$(pwd)

dataset="cora"
repeats=10

for ((i = 0; i < repeats; i++)); do
    python train/train_gnn.py --dataset "$dataset" --device $gpu_num --index "$i"

    python train/train_buffer.py \
        --dataset "$dataset" \
        --device $gpu_num \
        --index "$i" 
done

python evaluate/summarize.py --dataset "$dataset" --repeats "$repeats"

