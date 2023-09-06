import torch
import torch.nn as nn
import Utils  # Importing a custom module called Utils

# Define a custom LSTM model class
class LSTMModel(nn.Module):
    def __init__(self,
                 input_horizon: int,
                 hidden_size: int = Utils.LSTM_hidden_size,
                 num_layers: int = Utils.LSTM_NumLayer,
                 output_size: int = Utils.LSTM_outFeature,
                 DropOut: float = 0) -> None:
        super(LSTMModel, self).__init__()
        # Initialize the LSTM model with the specified parameters
        self.hidden_size = hidden_size  # Size of the hidden LSTM layer
        self.num_layers = num_layers    # Number of LSTM layers
        
        # Create an LSTM layer with batch-first setting and optional dropout
        self.lstm = nn.LSTM(input_horizon, hidden_size, num_layers, batch_first=True, )#dropout=DropOut
        
        # Create a fully connected (linear) layer to transform LSTM output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> tuple:
        # Forward pass through the LSTM layer
        out, (h_state, C_state) = self.lstm(x)
        
        # Check the shape of the 'out' tensor and apply the fully connected layer accordingly
        if   len(out.shape) == 2:
            # out = self.fc(out[:, :])  # If the shape is 2D, apply the linear layer to the entire sequence

            out = self.fc(h_state)
        elif len(out.shape) == 3:
            out = self.fc(out[:, -1, :])  # If the shape is 3D, apply the linear layer to the last time step
        
        return out  # Return the model's output
