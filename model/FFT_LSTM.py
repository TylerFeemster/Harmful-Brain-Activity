import numpy as np
import torch
from torch import nn, optim

class FFT_LSTM(nn.Module):
    def __init__(self, input_len, hidden_size, output_len, num_layers):
        super(FFT_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        hidden_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :])
        return out
    
    