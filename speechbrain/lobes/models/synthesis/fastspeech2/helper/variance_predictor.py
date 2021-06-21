"""
Variance Predictor Module

Authors
* Ayush Agarwal 2021
"""


import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm


class VariancePredictor(nn.Module):
    """
        Variance Predictor Module
    """

    def __init__(
        self,
        idim: int,
        n_layer: int = 2,
        n_chans: int = 384,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
    ):
        """ 
            Initialize Variance Predictor Module
        """
        super(VariancePredictor, self).__init__()

        self.conv = torch.nn.ModuleList()
        for index in range(n_layer):
            in_chans = idim if index == 0 else n_chans
            self.conv += [
                nn.Sequential(
                    nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size-1) // 2,
                        bias=bias,
                    ),
                    nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    nn.Dropout(dropout_rate)
                )
            ]
        self.linear = nn.Linear(n_chans, 1)
    

    def forward(self, xs , x_masks):

        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
        
        xs = self.linear(xs.transpose(1, 2))

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs