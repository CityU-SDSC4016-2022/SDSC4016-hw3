import torch
from torch import nn


class LSTMNet(nn.Module):
    def __init__(self, embedding: torch.Tensor, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: int, fix_embedding: bool):
        super(LSTMNet, self).__init__()
        # embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Whether fix embedding
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        self.lstm.flatten_parameters()
        inputs = self.embedding(inputs)
        x_var, _ = self.lstm(inputs)
        # dimension of x (batch, seq_len, hidden_size)
        x_var = x_var[:, -1, :]
        x_var = self.net(x_var)
        return x_var
