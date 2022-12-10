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
        # self.fc = nn.Linear(hidden_size, n_tokens)
        # self.attention = AttentionLayer(512)
        self.w_omega = nn.Parameter(torch.Tensor(
            hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.decoder = nn.Linear(hidden_size, n_tokens)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def __call__(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x, state)
        x = out.permute(1, 0, 2)
        # out = self.attention(out)
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        # feat = torch.sum(scored_x, dim=1)
        # feat形状是(batch_size, 2 * num_hiddens)
        outs = self.decoder(scored_x)
        # out形状是(batch_size, 2)
        # logit = self.fc(out)

        # return logit, (hn.detach(), cn.detach())
        return outs, (hn.detach(), cn.detach())
