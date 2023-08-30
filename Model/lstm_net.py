import torch
import torch.nn as nn
import Utils

class LSTMModel(nn.Module):
    def __init__(self,
                input_horizon   :int ,
                hidden_size     :int    = Utils.LSTM_hidden_size,
                num_layers      :int    = Utils.LSTM_NumLayer,
                output_size     :int    = Utils.LSTM_outFeature,
                DropOut         :float  = 0)->None:
        super(LSTMModel, self).__init__()
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        
        self.lstm           = nn.LSTM(input_horizon, hidden_size, num_layers, batch_first=True)
        self.fc             = nn.Linear(hidden_size, output_size)
        
    def forward(self, x)->tuple:
        out, (h_state,C_state) = self.lstm(x)
        
        if len(out.shape)==2:
            out = self.fc(out[:, :])
        if len(out.shape)==3:
            out = self.fc(out[:, -1, :])
        return out#,h_state,C_state