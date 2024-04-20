import os
import torch
import pickle
import numpy as np

#from https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def pad_and_mask(e, max_length, side='right'):
    length = len(e)
    tokens_pad = [0] * (max_length - length)
    mask = np.array([0] * max_length)
    if side == 'left':
        e = tokens_pad + e
        mask[max_length-length:] = 1
    else:
        e += tokens_pad
        mask[:length] = 1
    return e, mask

def write_cache(path, o):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, 'wb') as fo:
        pickle.dump(o, fo)
