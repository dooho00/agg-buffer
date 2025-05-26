#!/bin/bash
# This script runs training and evaluation for a specified dataset multiple times.
gpu_num=4 # Set the GPU number to use, change as needed

cd "$(dirname "$0")/.." 
export PYTHONPATH=$(pwd)

# Prompt user for the number of repetitions
read -p "Enter number of repetitions: " repeats

#"cora" "citeseer" "pubmed" "wiki_cs" "co_photo" "co_computer" "arxiv" "co_phy" "co_cs" "actor" "squirrel" "chameleon"
dataset="cora"

for ((i = 0; i < repeats; i++)); do
    python train/train_gnn.py --dataset "$dataset" --device $gpu_num --index "$i"

    python train/train_buffer.py \
        --dataset "$dataset" \
        --device $gpu_num \
        --index "$i" 
done

python evaluate/summarize.py --dataset "$dataset" --repeats "$repeats"