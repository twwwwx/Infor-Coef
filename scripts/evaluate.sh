
export TOKENIZERS_PARALLELISM=true
model_path=/mnt/data/user/bao_rong/tanwx/dynamic_struct/skimFT_COFI/bert-base-uncased_glue_mrpc_vanilla_lr2e-05_bsz32_epochs7_skim0.0_norm0.0004_entrophy0.0005_sparsity0.0/epoch0
task=mrpc
if [ "$task" == "mnli" ]; then
    label=3
else
    label=2
fi

python /root/infor_coef/evaluate.py \
    --dataset_name glue \
    --dataset_config_name ${task} \
    --model_name_or_path ${model_path} \
    --seed 42\
    --valid validation \
    --bsz 32\
    --num_labels ${label} \
    --eval_size 64 \
    --max_seq_length 128 \