# -*- coding:utf-8 -*-
import logging
import random
from datasets import load_dataset
from dataclasses import dataclass
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

name_to_path = {
    "imdb_domain": "/root/work1/data/imdb_domain/imdb_domain.py",
    "mnli_match_domain": "/root/work1/data/mnli_match_domain/mnli_match_domain.py",
    "qqp_domain": "/root/work1/data/qqp_domain/qqp_domain.py",
    "risk_adv": "/home/baorong/risk_adv/data/risk_adv.py",
    "adver_bart":"/root/adver_bart/attack/adver_bart.py"
}

task_to_keys = {
    "ag_news": ("text", None),
    "imdb": ("text", None),
    "imdb_domain": ("text", None),
    "risk_adv": ("text", None),
    "cola": ("sentence", None),
    "hans": ("premise", "hypothesis"),
    "mnli": ("premise", "hypothesis"),
    "mnli_match_domain": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "qqp_domain": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

MAX_SEQ_LEN = 256
logger = logging.getLogger(__name__)


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)




class Collator:
    """
    Collates transformer outputs.
    """

    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        model_inputs, labels = list(zip(*features))
        keys = list(model_inputs[0].keys())
        padded_inputs = {}
        for key in keys:
            if key == 'input_ids':
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            sequence = [x[key] for x in model_inputs]
            sequence = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = sequence
        labels = torch.tensor(labels)
        return padded_inputs, labels



class Robust_dataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            split='train'
    ):
        if args.dataset_name in name_to_path.keys():
            if args.dataset_config_name is not None:
                self.raw_dataset = load_dataset(name_to_path[args.dataset_name], args.dataset_config_name)
            else:
                self.raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name)
        else:
            if args.dataset_config_name is not None:
                self.raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name)
            else:
                self.raw_dataset = load_dataset(args.dataset_name)
        self.dataset = self.raw_dataset[split]
        self.sub_datasets = list(self.raw_dataset.column_names)
        self.column_names = self.raw_dataset[self.sub_datasets[0]].column_names
        if args.dataset_config_name is not None:
            self.input_columns = task_to_keys[args.dataset_config_name]
        else:
            self.input_columns = task_to_keys[args.dataset_name]
        self.key1 = self.input_columns[0]
        self.key2 = self.input_columns[1]
        self.label_column_name = 'label'
        # if args.task == 'NER':
        #     self.label_column_name = 'ner_tags'
        self.tokenizer = tokenizer
        self.model_name_or_path = args.model_name_or_path if hasattr(args, 'model_name_or_path') else None
        self.padding = 'max_length' if args.pad_to_max_length else False
        self.max_length = args.max_seq_length if hasattr(args, 'max_seq_length') else MAX_SEQ_LEN

    def _format_examples(self, examples):
        """
        Only for some task which has ONE input column, such as SST-2 and IMDB, NOT work for NLI such as MRPC.
        """
        texts = ((examples[self.key1],) if self.key2 is None else (examples[self.key1], examples[self.key2]))
        inputs = self.tokenizer(*texts, truncation=True, max_length=self.max_length, return_tensors='pt',padding=self.padding)
        labels = int(examples[self.label_column_name])
        return inputs, labels


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """Return i-th sample."""
        if isinstance(i, int):
            return self._format_examples(self.dataset[i])
        else:
            return [
                self._format_examples(self.dataset[j]) for j in range(i.start, i.stop)
            ]
class ExponentialMovingAverage:
    def __init__(self, beta=0.99):
        self._beta = beta
        self._x = 1e-7

    def reset(self):
        self._x = 1e-7
        
    def update(self, x):
        self._x = self._beta * self._x + (1-self._beta)*x

    def get_metric(self):
        return self._x
    
    
def linear_flops(input_dim, output_dim,bias=True):
    flops = 0
    flops += input_dim * output_dim
    return flops

@dataclass
class data_args:
    dataset_name: str = 'glue'
    dataset_config_name: str = 'qnli'
    valid: str = 'validation'
    max_seq_length: int = 128
    pad_to_max_length: bool = False
    bsz: int = 32

