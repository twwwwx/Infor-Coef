from dataclasses import dataclass
import torch
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple

from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput

@dataclass
class BaseModelOutputSkim(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None
    soft_mask: Optional[torch.FloatTensor] = None



@dataclass
class BaseModelOutputWithPoolingSkim(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None
    soft_mask: Optional[torch.FloatTensor] = None


@dataclass
class SequenceClassifierOutputSkim(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None
    skim_loss: Optional[torch.FloatTensor] = None
    classification_loss: Optional[torch.FloatTensor] = None
    tokens_remained: Optional[torch.FloatTensor] = None
    layer_tokens_remained: Optional[Tuple[torch.FloatTensor]] = None
    soft_mask: Optional[torch.FloatTensor] = None

def masked_softmax(vec, mask, dim=1, eps=1e-7):
    # vec = vec - vec.max(dim=dim,keepdim=True)[1]
    vec = vec.clamp(min=-50,max=50)
    mask = mask[:,None,None,:]
    exps = torch.exp(vec) + eps
    masked_exps = exps * mask.float() 
    masked_sums = masked_exps.sum(dim, keepdim=True) 
    return (masked_exps/masked_sums)
def convert_softmax_mask_to_digit(skim_mask):
    # skim_mask [batch, from, to, seq_len]
    return (skim_mask == 0).to(dtype=torch.int64).unsqueeze(1).unsqueeze(1)

def trunc_with_mask_batched(input, mask, dim):
    """
    trunc a batched input at dim
        e.g. hidden_states ([batch, seq_len, hidden_size])
            attention_mask ([batch, layer, head, seq_len])
    mask: [batch, seq_len]
    """
    assert input.shape[dim]==mask.shape[1]

    if dim != 1:
        input = input.transpose(1, dim)

    transpose_shape = list(input.shape)
    transpose_shape[1] = -1

    trunc_input = input[mask].view(transpose_shape)

    if dim != 1:
        trunc_input = trunc_input.transpose(1, dim)

    return trunc_input
