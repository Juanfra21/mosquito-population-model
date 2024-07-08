import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size  # Store the hidden size as an instance variable
        self.num_layers = (num_layers)  # Store the number of LSTM layers as an instance variable
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # Define the LSTM layer
        self.fc = nn.Linear(hidden_size, output_size)  # Define the fully connected output layer

    def forward(self, x):
        # Initialize hidden state and cell state with correct dimensions for unbatched input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state

        # Forward pass through LSTM
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))  # Add an extra dimension for batch (batch_first=True)

        # Only take the output from the last time step
        out = self.fc(out[:, -1, :])  # Pass the last time step output through the fully connected layer

        return out  # Return the final output
