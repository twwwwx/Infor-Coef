#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
# this is used for skim-training the pruned CoFi

dir=/mnt/data/user/bao_rong/tanwx

TASK=$2
# if task==mrpc, label=2; if task==mnli, label=3; if task==qnli, label=2; if task==sst2, label=2
if [ "$TASK" == "mnli" ]; then
    label=3
    valid=validation_matched
else
    label=2
    valid=validation
fi

NORM=$3
ENTRO=$4
EPOCH=$5
LR=$6
model_name_or_path=$7
sparsity=$8
bsz=$9
skim=${10}

python run_glue_vanilla.py \
    --dataset_name glue \
    --dataset_config_name ${TASK} \
    --model_name_or_path ${model_name_or_path} \
    --valid ${valid} \
    --num_labels ${label} \
    --bsz ${bsz} \
    --eval_size 64 \
    --epochs ${EPOCH} \
    --lr ${LR} \
    --skim_coefficient ${skim} \
    --norm_coefficient ${NORM} \
    --entrophy_coefficient ${ENTRO}\
    --sparsity ${sparsity} \
    --output_dir ${dir}/output \
    --eval_steps 500 \

