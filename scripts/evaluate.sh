
export TOKENIZERS_PARALLELISM=true
model_path=/root/infor_coef/model # modify this path to your model path
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