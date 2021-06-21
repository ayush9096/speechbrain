"""
Length-Regulator Module for Feed-Forward Transformers

Authors
* Ayush Agarwal 2021
"""


import torch
from torch import nn
from torch.nn import functional as F
import logging


# Helper Function 

def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad



class LengthRegulator(nn.Module):

    """
        This is module of length regulator described in FastSpeech
    """

    def __init__(self, pad_value=0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value
    
    def forward(self, xs, ds, alpha=1.0):
        """
            It expands character or phoneme-level embedding feature to frame level by repeating each
            feature based on the corresponding duration prediction
        """

        if alpha != 1.0:
            assert alpha > 0
            ds = torch.round(ds.float() * alpha).long()
        
        if ds.sum() == 0:
            logging.warning(" Predicted Durations includes all 0 sequences")
            ds[ds.sum(dim=1).eq(0)] = 1
        
        return pad_list([self.repeat_fn(x, d) for x, d in zip(xs, ds)], self.pad_value)

    
    def repeat_fn(self, x, d):
        """
            Repeat Each Frame According to the Durations
        """

        return torch.cat(
            [x_.repeat(int(d_), 1) for x_,d_ in zip(x, d) if d_ != 0], dim=0
        )