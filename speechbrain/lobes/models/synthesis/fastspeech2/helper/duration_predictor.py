"""
Duration Predictor Module

Authors
* Ayush Agarwal 2021
"""


import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm


class DurationPredictor(nn.Module):

    """
        Duration Predictor Module
        This is a module of duration predictor described in 
        `FastSpeech: Fast, Robust and Controllable Text-to-Speech`
        
        The duration predictor predicts a duration of each frame from the hidden embeddings of encoder
    """

    def __init__(
        self,
        idim,
        n_layers=2,
        n_chans=384,
        kernel_size=3,
        dropout_rate=0.1,
        offset=1.0
    ):
        """ 
            Initialize Duration Predictor Module

            Args:
                idim (int): Input Dimension
                n_layers (int, optional): Number of Convolutional Layers
                n_chans (int, optional): Number of channels of Convolutional Layers
                kernel_size (int, optional): Kernel Size of Convolutional Layers
                dropout_rate (float, optional): Dropout Rate
                offset (float, optional): Offset value to avoid nan in log domain
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for index in range(n_layers):
            in_chans = idim if index == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size-1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate)
                )
            ]
        self.linear = torch.nn.Linear(n_chans, 1)
    

    def _forward(self, xs, x_masks=None, is_inference=False):

        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
        
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)

        if is_inference:
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long() # Avoid Negative Values
        
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        
        return xs



    def forward(self, xs, x_masks=None):
        """
            Calculate Forward Propogation

            Args:
                xs (Tensor) : Batch Of Input Sequences (B, Tmax, idim)
                x_masks (ByteTensor, optional) : Batch of masks indicating padded part (B, Tmax)
            
            Returns:
                Tensor: Batch of predicted durations in log domain (B, Tmax)
        """

        return self._forward(xs, x_masks, False)

    
    def inference(self, xs, x_masks=None):
        
        return self._forward(xs, x_masks, True)




class DurationPredictorLoss(nn.Module):
    """
        Loss Function Module for Duration Predictor
    """

    def __init__(self, offset=1.0, reduction="mean"):
        super(DurationPredictorLoss, self).__init__()

        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    
    def forward(self, outputs, targets):
        """
            Calculates Duration Predictor Loss
        """

        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss


        

