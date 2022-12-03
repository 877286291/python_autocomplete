from typing import Optional, Tuple

import torch
from labml_nn.lstm import LSTM
from torch import nn
from torch.nn import functional as F

from models import AutoregressiveModel


class AttentionLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs):
        inputs = torch.permute(inputs, (1, 0, 2))
        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs).permute(0, 2, 1)  # 先进行一次转置
        V = self.V_linear(inputs)
        alpha = torch.matmul(Q, K)
        alpha = F.softmax(alpha, dim=2)
        return torch.matmul(alpha, V)


class LstmModel(AutoregressiveModel):
    def __init__(self, *,
                 n_tokens: int,
                 embedding_size: int,
                 hidden_size: int,
                 n_layers: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(n_tokens, embedding_size)
        self.lstm = LSTM(input_size=embedding_size,
                         hidden_size=hidden_size,
                         n_layers=n_layers)
        self.fc = nn.Linear(hidden_size, n_tokens)
        self.attention = AttentionLayer(512)

    def __call__(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x, state)
        out = self.attention(out)
        logit = self.fc(out)

        return logit, (hn.detach(), cn.detach())
