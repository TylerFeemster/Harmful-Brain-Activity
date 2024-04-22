import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F



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

class Model(nn.Module):
    def __init__(self, ti, th, to, tlayers, fi, fh, fo, flayers):
        super(Model, self).__init__()
        self.t_hidden = th
        self.t_layers = tlayers
        self.t_lstm = nn.LSTM(ti, th, tlayers, batch_first=True)
        self.t_output = nn.Linear(th, to)

        self.f_hidden = fh
        self.f_layers = flayers
        self.f_lstm = nn.LSTM(fi, fh, flayers, batch_first=True)
        self.f_output = nn.Linear(fh, fo)

        self.combine = nn.Linear()

    def forward(self, t, f):
        t_hstates = torch.zeros(self.t_layers, t.size(0), self.t_hidden)
        t_cstates = torch.zeros(self.t_layers, t.size(0), self.t_hidden)
        t_out, _ = self.t_lstm(t, (t_hstates, t_cstates))
        t_out = self.t_output(t_out[:, -1, :])

        f_hstates = torch.zeros(self.f_layers, f.size(0), self.f_hidden)
        f_cstates = torch.zeros(self.f_layers, f.size(0), self.f_hidden)
        f_out, _ = self.f_lstm(f, (f_hstates, f_cstates))
        f_out = self.f_output(f_out[:, -1, :])

        







