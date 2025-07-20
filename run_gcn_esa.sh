#!/bin/bash

# Train GCN+ESA model on PDBBind dataset

python train_gcn_esa.py \
    --train_set datasets/PDBBind/processed/identity30/train.pkl \
    --valid_set datasets/PDBBind/processed/identity30/valid.pkl \
    --test_set datasets/PDBBind/processed/identity30/test.pkl \
    --save_dir ./results/gcn_esa \
    --hidden_size 128 \
    --gcn_layers 2 \
    --esa_layers 3 \
    --num_heads 8 \
    --dropout 0.1 \
    --lr 0.001 \
    --batch_size 16 \
    --max_epoch 100 \
    --patience 20 \
    --gpus 0 \
    --seed 42
