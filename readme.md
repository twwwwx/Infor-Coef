# Infor-Coef
## Overview
The prevalence of Transformer-based pre-trained language models (PLMs) has led to their wide adoption for various natural language processing tasks. However, their excessive overhead leads to large latency and computational costs. The statically compression methods allocate fixed computation to different samples, resulting in redundant computation. The dynamic token pruning method selectively shortens the sequences but are unable to change the model size and hardly achieve the speedups as static pruning. In this paper, we propose a model accelaration approaches for large language models that incorporates dynamic token downsampling and static pruning, optimized by the information bottleneck loss. Our model, Infor-Coef, achieves an 18x FLOPs speedup with an accuracy degradation of less than 8\% compared to BERT. This work provides a promising approach to compress and accelerate transformer-based models for NLP tasks.

## Run Infor-Coef

1. Create a conda virtual environment and activate it
```
conda create --name infor_coef --file requirements.txt
conda activate infor_coef
```
2. Download the pruned [CoFi](https://github.com/princeton-nlp/CoFiPruning) models (or train it from scratch)
3. Modify the training parameters in `action.sh` and run it.

### Training Parameters:
Our script supports only sigle-GPU training. The parameters are as follows:

 - `TASK`: the task to train, including `mrpc`,`sst-2`, `mnli`, `qnli`
 - `sparsity`: the sparsity of the pruned model. 
 - `model_name_or_path`: the path of the pruned model.
 - `CUDA`: the GPU id to use
 - `NORM`: the norm-based penalty parameter of the information bottleneck loss
 - `ENTRO`: the entropy regularization parameter of the information bottleneck loss
 - `EPOCHs`: the number of training epochs
 - `LR`: the learning rate
 - `bsz`: the batch size
 - `skim`: the hyper-parameter of the dynamic token skim in [Transkimmer](https://github.com/ChandlerGuan/Transkimmer). Set to 0 in our model.

### Evaluate Infor-Coef

Run `eval.sh` to evaluate the pruned model on the corresponding task. The parameters are the same as `action.sh`.