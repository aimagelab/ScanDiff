import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Encoder(nn.Module):
    def __init__(self, encoder_dimensions: List[int]):
        super(Encoder, self).__init__()
        self.fcs = nn.ModuleList()
        
        num_dim = len(encoder_dimensions) - 1
        for i in range(num_dim):
            self.fcs.append(nn.Linear(encoder_dimensions[i], encoder_dimensions[i+1]))
            if i < num_dim - 1:
                self.fcs.append(nn.BatchNorm1d(encoder_dimensions[i+1]))
                self.fcs.append(nn.ReLU())

    def forward(self, x):
        for layer in self.fcs:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x.transpose(1,2)).transpose(1, 2)
            else:
                x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, decoder_dimensions: List[int]):
        super(Decoder, self).__init__()
        self.fcs = nn.ModuleList()
        
        num_dim = len(decoder_dimensions) - 1
        for i in range(num_dim):
            self.fcs.append(nn.Linear(decoder_dimensions[i], decoder_dimensions[i+1]))
            
            if i < num_dim - 1:
                self.fcs.append(nn.BatchNorm1d(decoder_dimensions[i+1]))
                self.fcs.append(nn.ReLU())

    def forward(self, x):
        for layer in self.fcs:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x.transpose(1,2)).transpose(1, 2)
            else:
                x = layer(x)
        return x