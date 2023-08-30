CUDAs=4
TASK=mrpc ##
sparsity=0.0 ##
model_name_or_path=/root/infor_coef/model/MRPC60
CUDA=0
NORM=4e-4
ENTRO=4e-4
EPOCHs=7
LR=2e-5
bsz=32
skim=0.0

bash scripts/runglue_skim.sh ${CUDA} ${TASK} ${NORM} ${ENTRO} ${EPOCHs} ${LR} ${model_name_or_path} ${sparsity} ${bsz} ${skim}


