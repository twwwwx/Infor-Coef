# -*- coding:utf-8 -*-
"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import os
# os.environ["HF_HOME"] = "/root/huggingface"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
from pathlib import Path
import random
import numpy as np
# from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,AutoModel
)
from sklearn.metrics import confusion_matrix,cohen_kappa_score,f1_score
import utils.utils_data as utils
from utils.struct_utils import calculate_parameters,load_zs,load_model
from module.modeling_inforcoef import CoFiBertForSequenceClassification as InforCoefForSequenceClassification
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--dataset_config_name", default=None, type=str)
    parser.add_argument('--output_dir', type=Path, default=Path('/root/tmp/saved_models/'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--task', type=str, default='nli')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_lower_case', type=bool, default=True)

    # hyper-parameters
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--skim_coefficient', type=float, default=0, help='skim coefficient')
    parser.add_argument('--norm_coefficient', type=float, default=.0, help='norm coefficient')
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument('--max_seq_length', type=int, default=128, help='max sequence length')
    parser.add_argument('--need_zs', action='store_true', help='need zs')
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = '.'
    return args


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args):
    set_seed(args.seed)
    logger.info(args)
    output_dir = args.model_name_or_path if os.path.exists(args.model_name_or_path) else args.output_dir

    log_file = os.path.join(output_dir, 'eval_INFO.log')
    logger.addHandler(logging.FileHandler(log_file))

    # pre-trained model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(args.model_name_or_path):
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        config.skim_coefficient = args.skim_coefficient
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = InforCoefForSequenceClassification.from_pretrained(args.model_name_or_path,)
    else:
        model = InforCoefForSequenceClassification.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model.config.skim_coefficient = args.skim_coefficient
        config = model.config

    model.to(device)

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    test_dataset = utils.Robust_dataset(args,tokenizer,split=args.valid)
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator, num_workers=2)

    # Log a few random samples from the training set:
    
    for index in random.sample(range(len(test_dataset)),1):
        logger.info(f"Sample {index} of the training set: {test_dataset[index]}.")

    logger.info('Evaluating...')
    model.eval()

    with torch.no_grad():
        y_gold = []
        y_pred = []
        all_skim_loss, all_tokens_remained = list(), list()
        all_layer_tokens_remained = [[] for _ in range(config.num_hidden_layers)]
        model_base = InforCoefForSequenceClassification.from_pretrained('bert-base-uncased')
        for model_inputs, labels in test_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            # get batch size and length from attn mask
            batch_size, seq_len = model_inputs['attention_mask'].shape
            y_gold.extend(labels.tolist())
            outputs = model(**model_inputs)
            logits = outputs.logits
            _, preds = logits.max(dim=-1)
            y_pred.extend(preds.cpu().tolist())


            all_skim_loss.append(outputs.skim_loss)
            all_tokens_remained.append(outputs.tokens_remained)
            for layer_idx,mac in enumerate(outputs.layer_tokens_remained):
                all_layer_tokens_remained[layer_idx].append(mac)

        all_layer_tokens_remained = [torch.mean(torch.stack(layer_tokens_remained)).cpu().item() for layer_tokens_remained in all_layer_tokens_remained]
        kappa_matrix = confusion_matrix(y_gold,y_pred)
        f1 = f1_score(y_gold,y_pred,average='binary') if args.num_labels == 2 else f1_score(y_gold,y_pred,average='macro')
        accuracy = sum(y_gold[i] == y_pred[i] for i in range(len(y_gold))) / len(y_gold)

        # params
        params = calculate_parameters(model)
        model_base = AutoModel.from_pretrained('bert-base-uncased')
        base_params = calculate_parameters(model_base)
        param_reduction = base_params / params


        
        logger.info(f'Matrix: {kappa_matrix}, '
                    f'Score: {f1}, '
                    f'Accuracy: {accuracy}\n'
                    f'params:{params},'
                    f'param_reduction:{param_reduction}x,\n'
                    f'skim_loss:{torch.mean(torch.stack(all_skim_loss))},'
                    f'tokens_remained:{torch.mean(torch.stack(all_tokens_remained))},\n'
                    f'layer_tokens_remained:{all_layer_tokens_remained}')
            

if __name__ == '__main__':

    args = parse_args()
    main(args)