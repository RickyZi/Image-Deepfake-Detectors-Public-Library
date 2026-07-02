#!/usr/bin/env bash

# script test

# python ./detectors/R50_nodown/optuna_search.py \
#     --study_name R50nd_optunaTEST \
#     --mode standard \
#     --n_trials 3 \
#     --pretrained_ckpt ./detectors/R50_nodown/checkpoint/pretrained/weights/best.pt \
#     --data_root ./truefake_2k/tf2k_lr_org/style/vintage_VN01 \
#     --split_file ./test2k_splits.json \
#     --max_epochs 5 \
#     --patience 3 \
#     --num_threads 4

# search best hyperparam for 2 worst performing models (VN01 and BW01)
# if results are good, we may try to extend this to all datasets for R50_nd

python ./detectors/R50_nodown/optuna_search.py \
    --study_name R50_nd_BW01_standard \
    --mode standard \
    --n_trials 30 \
    --pretrained_ckpt ./detectors/R50_nodown/checkpoint/pretrained/weights/best.pt \
    --data_root ./truefake_2k/tf2k_lr_org/style/bw_BW01 \
    --split_file ./test2k_splits.json \
    --max_epochs 20 \
    --patience 5 \
    --batch_size 64 \
    --num_threads 4

