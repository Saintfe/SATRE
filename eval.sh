#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python eval.py saved_models/$2/ --dataset $3
