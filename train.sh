#!/bin/bash
mkdir -p ./saved_models/$2/
touch ./saved_models/$2/compare.$2.txt
CUDA_VISIBLE_DEVICES=$1  python train.py --id $2 --seed 0  --num_epoch 100 --pooling max  --log_step 100 --save_epoch 101

