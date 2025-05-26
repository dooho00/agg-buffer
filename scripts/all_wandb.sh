#!/bin/bash
# This script runs training and evaluation across all datasets and logs results to Weights & Biases (wandb).
gpu_num=4 # Set the GPU number to use, change as needed

cd "$(dirname "$0")/.." 
export PYTHONPATH=$(pwd)

# Prompt user for the number of repetitions
read -p "Enter number of repetitions: " repeats

dataset=("cora" "citeseer" "pubmed" "wiki_cs" "co_photo" "co_computer" "arxiv" "co_phy" "co_cs" "actor" "squirrel" "chameleon")

# Loop over each dataset and repetition
for dataset in "${dataset[@]}"; do
    for ((i = 0; i < repeats; i++)); do

        python train/train_gnn.py --dataset "$dataset" --device $gpu_num --index "$i"

        python train/train_buffer.py \
            --dataset "$dataset" \
            --device $gpu_num \
            --index "$i" \
            --wandb true
    done
done
