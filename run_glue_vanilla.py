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
import math
from pathlib import Path
import random
import numpy as np
# from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
from sklearn.metrics import confusion_matrix,cohen_kappa_score,f1_score
import utils.utils_data as utils
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
    parser.add_argument('--eval_steps', type=int, default=1000)

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
    parser.add_argument('--max_steps', type=int, default=0, help='max steps')
    parser.add_argument('--skim_coefficient', type=float, default=0.2, help='skim coefficient')
    parser.add_argument('--norm_coefficient', type=float, default=4e-4, help='norm coefficient')
    parser.add_argument('--entrophy_coefficient', type=float, default=5e-4, help='entrophy coefficient')
    parser.add_argument('--max_seq_length', type=int, default=128, help='max sequence length')
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument('--sparsity', type=float, default=0.6, help='sparsity')
    # freeze_skim store true
    parser.add_argument('--freeze_skim', action='store_true', help='freeze skim')
    parser.add_argument('--freeze_skim_epoch', type=int, default=0, help='freeze skim epoch')
    parser.add_argument('--warmup_epochs',type=int,default=1)


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


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def neg_entrophy(logits):
    entrophy = 0
    l,r = 1e-6,1-1e-6
    logits = logits.clamp(min=l,max=r)
    # plnp + (1-p)ln(1-p)
    entrophy = logits*torch.log(logits)+(1-logits)*torch.log(1-logits[0])
    return entrophy.mean()

def compute_loss(model,batch_inputs,labels,args,global_step,epoch):
    outputs = model(**batch_inputs,labels=labels,output_hidden_states=True)
    logits = outputs.logits
    num_hidden_layers = model.config.num_hidden_layers
    cls_loss = F.cross_entropy(logits, labels)
    loss = cls_loss
    norm_loss,entrophy_loss,skim_loss = 0.,0.,0.
    if args.skim_coefficient > 0 and epoch >= args.warmup_epochs:
        skim_loss = outputs.skim_loss
        loss = loss + args.skim_coefficient * skim_loss
    if  args.norm_coefficient > 0 and epoch >= args.warmup_epochs:
        all_hidden_states = outputs.hidden_states
        for i,hidden_states in enumerate(all_hidden_states):
            # soft_mask : 12x([B,L]), hidden: [B,L,D]
            if i == len(outputs.soft_mask):
                break
            pruned_hidden = hidden_states * outputs.soft_mask[i].unsqueeze(-1)
            norm_loss = norm_loss + torch.mean(torch.norm(pruned_hidden,p='fro'))
        norm_loss = norm_loss / num_hidden_layers
        # norm_coefficient = min(args.norm_coefficient,global_step*args.norm_coefficient / 3000)
        loss = loss + args.norm_coefficient * norm_loss
    if args.entrophy_coefficient > 0:
        all_hidden_states = outputs.hidden_states
        for i,soft_mask in enumerate(outputs.soft_mask):
            entrophy_loss = entrophy_loss + neg_entrophy(soft_mask)
        entrophy_loss = entrophy_loss / num_hidden_layers
        # entrophy_coefficient = args.entrophy_coefficient * (1 - math.exp(-global_step / 1000))
        entrophy_coefficient = args.entrophy_coefficient
        loss = loss + entrophy_coefficient * entrophy_loss
    loss_items = {'cls_loss': cls_loss, 'skim_loss': skim_loss, 'norm_loss': norm_loss, 'entrophy_loss': entrophy_loss}
    return loss_items,loss,logits

def evaluate(config,test_loader,model,device,epoch,logger):
    logger.info(f"*evaluation in the middle of epoch{epoch}")
    model.eval()
    with torch.no_grad():
        y_gold = []
        y_pred = []
        all_skim_loss, all_tokens_remained = list(), list()
        for model_inputs, labels in test_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            # get batch size and length from attn mask
            y_gold.extend(labels.tolist())
            outputs = model(**model_inputs)
            logits = outputs.logits
            _, preds = logits.max(dim=-1)
            y_pred.extend(preds.cpu().tolist())


            all_skim_loss.append(outputs.skim_loss)


        kappa_matrix = confusion_matrix(y_gold,y_pred)
        f1 = f1_score(y_gold,y_pred,average='macro')
        accuracy = sum(y_gold[i] == y_pred[i] for i in range(len(y_gold))) / len(y_gold)

        
    logger.info(f'Matrix: {kappa_matrix}, '
                    f'Score: {f1}, '
                    f'Accuracy: {accuracy}\n'
                    f'skim_loss:{torch.mean(torch.stack(all_skim_loss))},')


def main(args):
    set_seed(args.seed)
    logger.info(args)
    model_name = 'bert-base-uncased' if os.path.exists(args.model_name_or_path) else 'bert-origin'
    if args.dataset_name != 'glue':
        output_dir = Path(os.path.join(args.output_dir, '{}_{}_vanilla_lr{}_bsz{}_epochs{}_skim{}_norm{}_entrophy{}'
                                       .format(model_name,args.dataset_name, args.lr, args.bsz,
                                               args.epochs,args.skim_coefficient,args.norm_coefficient,args.entrophy_coefficient)))
    else:
        output_dir = Path(os.path.join(args.output_dir, '{}_{}_{}_vanilla_lr{}_bsz{}_epochs{}_skim{}_norm{}_entrophy{}_sparsity{}_freeze{}'
                                       .format(model_name,args.dataset_name,args.dataset_config_name,
                                               args.lr, args.bsz, args.epochs,args.skim_coefficient,args.norm_coefficient,args.entrophy_coefficient,args.sparsity,args.freeze_skim_epoch)))
    if not output_dir.exists():
        logger.info(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')
    log_file = os.path.join(output_dir, 'INFO.log')
    logger.addHandler(logging.FileHandler(log_file))

    # pre-trained model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(args.model_name_or_path):
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        config.skim_coefficient = args.skim_coefficient
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = InforCoefForSequenceClassification.from_pretrained(args.model_name_or_path)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        model = InforCoefForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model.config.skim_coefficient = args.skim_coefficient

    model.to(device)

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    # for training
    dataset = utils.Robust_dataset(args,tokenizer,split='train')
    train_size = int(0.9*len(dataset))
    dev_size = len(dataset) - train_size
    train_dataset, dev_dataset = torch.utils.data.random_split(dataset,[train_size,dev_size])
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator, num_workers=2)
    # for validation and test
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator, num_workers=2)
    test_dataset = utils.Robust_dataset(args,tokenizer,split=args.valid)
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator, num_workers=2)

    # for i, (model_inputs, labels) in enumerate(train_loader):
    #     if i > 3:
    #         break
    #     logger.info(model_inputs['input_ids'])
    # exit(0)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_grouped_parameters_wo_skim = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'skim' not in n],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'skim' not in n], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters_wo_skim,lr=args.lr) if args.freeze_skim else torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    best_test_epoch = -1
    best_test_acc = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        logger.info('Training...')
        model.train()
        avg_loss = utils.ExponentialMovingAverage()
        total_loss = 0.0
        if epoch == args.epochs - args.freeze_skim_epoch:
            logger.info('Freezing skim layer...')
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters_wo_skim, lr=args.lr)
        for i,(model_inputs, labels) in enumerate(train_loader):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            model.zero_grad()
            # outputs = model(**model_inputs,labels=labels,output_hidden_states=True,output_attentions=True)
            # loss = outputs.loss
            loss = compute_loss(model, model_inputs, labels,args,global_step,epoch)[1]
            # loss = losses[1]

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss.update(total_loss)
            global_step += 1
            if args.max_steps > 0 and i > args.max_steps:
                logger.info(f'avg_loss:{avg_loss.get_metric()}'
                            f'loss:{loss.item()}'
                            f'global_step:{global_step}'
                            f'logits:{outputs.logits}')
                break
            
            if global_step % args.eval_steps == 0:
                evaluate(config, test_loader,model, device, epoch, logger)
            
        if epoch // 5 == 0:
            s = Path(str(output_dir) + '/epoch' + str(epoch))
            if not s.exists():
                s.mkdir(parents=True)
            model.save_pretrained(s)
            tokenizer.save_pretrained(s)


        logger.info('Evaluating...at global_step:{}'.format(global_step))
        model.eval()

        with torch.no_grad():
            y_gold = []
            y_pred = []
            for model_inputs, labels in dev_loader:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                y_gold.extend(labels.tolist())
                logits = model(**model_inputs).logits
                _, preds = logits.max(dim=-1)
                y_pred.extend(preds.cpu().tolist())
            kappa_matrix = confusion_matrix(y_gold,y_pred)
            kappa_sacore = cohen_kappa_score(y_gold, y_pred)
            accuracy = sum(y_gold[i] == y_pred[i] for i in range(len(y_gold))) / len(y_gold)
            logger.info(f'Dev set, Epoch: {epoch}, '
                        f'Loss: {avg_loss.get_metric(): 0.4f}, '
                        f'Matrix: {kappa_matrix}, '
                        f'Score: {kappa_sacore}, '
                        f'Accuracy: {accuracy}')

        with torch.no_grad():
            y_gold = []
            y_pred = []
            all_skim_loss, all_tokens_remained = list(), list()
            all_layer_tokens_remained = [[] for _ in range(config.num_hidden_layers)]
            for model_inputs, labels in test_loader:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
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
            kappa_sacore = cohen_kappa_score(y_gold, y_pred)
            accuracy = sum(y_gold[i] == y_pred[i] for i in range(len(y_gold))) / len(y_gold)
            f1 = f1_score(y_gold, y_pred, average='binary') if args.num_labels == 2 else f1_score(y_gold, y_pred, average='macro')

            if accuracy > best_test_acc:
                best_test_acc = accuracy
                best_test_epoch = epoch
            logger.info(f'Test set, Epoch: {epoch}, '
                        f'Loss: {avg_loss.get_metric(): 0.4f}, '
                        f'Matrix: {kappa_matrix}, '
                        f'f1 Score: {f1}, '
                        f'Accuracy: {accuracy}\n'
                        f'skim_loss:{torch.mean(torch.stack(all_skim_loss))},'
                        f'tokens_remained:{torch.mean(torch.stack(all_tokens_remained))},'
                        f'layer_tokens_remained:{all_layer_tokens_remained}\n')
            
    logger.info(f'Best Test Epoch:{best_test_epoch}, Best Test Acc:{best_test_acc}')
    logger.info(f'norm_coefficient:{args.norm_coefficient},'
                f'entropy_coefficient:{args.entrophy_coefficient},'
                f'lr:{args.lr}')

if __name__ == '__main__':

    args = parse_args()
    main(args)
