import torch
import torch.nn as nn

class ConditionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ConditionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden)
        
        # out: (batch, seq_len, hidden_size)
        out = self.fc(out)
        
        # out: (batch, seq_len, output_size)
        return out, hidden
