#!/usr/bin/env bash

# create folders if missing
[ ! -d "results/" ] && mkdir -p "results/"
[ ! -d "plots/" ] && mkdir -p "plots/"

# TS
settings="densenet40_c10 resnet110_SD_c100 densenet40_c100 densenet161_imgnet lenet5_c10 lenet5_c100 resnet_wide32_c10 resnet_wide32_c100 resnet50_birds resnet110_c10 resnet110_c100 resnet110_SD_c10 resnet152_imgnet resnet152_SD_SVHN"
for setting in $settings
do
  python3 experiments.py \
    --logits_path "logits/probs_${setting}_logits.p" \
    --setting "${setting}" \
    --method "TS" \
    --save_file "results/results_TS.csv" \
    --start_rep 10000
done

# ETS
settings="densenet40_c10 resnet110_SD_c100 densenet40_c100 densenet161_imgnet lenet5_c10 lenet5_c100 resnet_wide32_c10 resnet_wide32_c100 resnet50_birds resnet110_c10 resnet110_c100 resnet110_SD_c10 resnet152_imgnet resnet152_SD_SVHN"
for setting in $settings
do
  python3 experiments.py \
    --logits_path "logits/probs_${setting}_logits.p" \
    --setting "${setting}" \
    --method "ETS" \
    --save_file "results/results_ETS.csv" \
    --start_rep 10000
done

# DIAG
settings="resnet50_nts_birds resnet101_cars resnet101pre_cars resnet50pre_cars densenet40_c10 resnet110_c10 resnet_wide32_c10 densenet40_c100 resnet110_c100 resnet_wide32_c100 densenet161_imgnet pnasnet5_large_imgnet resnet152_imgnet resnet152_sd_svhn"
for setting in $settings
do
  python3 experiments.py \
    --logits_path "logits/diag_${setting}" \
    --setting "${setting}" \
    --method "DIAG" \
    --save_file "results/results_DIAG.csv" \
    --start_rep 10000
done
