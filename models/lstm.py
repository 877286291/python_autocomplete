from typing import Optional, Tuple

import torch
from labml_nn.lstm import LSTM
from torch import nn

from models import AutoregressiveModel


class LstmModel(AutoregressiveModel):
    def __init__(self, *,
                 n_tokens: int,
                 embedding_size: int,
                 hidden_size: int,
                 n_layers: int,
                 round_: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.round_ = round_
        self.embedding = nn.Embedding(n_tokens, embedding_size)
        self.lstm = LSTM(input_size=embedding_size,
                         hidden_size=hidden_size,
                         n_layers=n_layers)
        self.fc = nn.Linear(hidden_size, n_tokens)

    @staticmethod
    def _linear(args, output_size):
        total_arg_size = 0
        shapes = [a.shape for a in args]
        for shape in shapes:
            total_arg_size += shape[-1]
        weights = nn.Parameter(torch.Tensor(total_arg_size, output_size))
        if torch.cuda.is_available():
            weights.cuda()
        if len(args) == 1:
            res = torch.matmul(args[0], weights)
        else:
            res = torch.matmul(torch.reshape(args, (-1, total_arg_size)), weights)
        return res

    def _do_feature_masking(self, x, h, rounds, batch_first=False):
        if batch_first:
            bs, seq_sz, _ = x.size()
        else:
            seq_sz, bs, _ = x.size()
        for _round in range(rounds):
            if _round % 2 == 0:
                x = 2 * torch.sigmoid(self._linear(h.reshape(bs, -1), self.embedding_size)) * x
            else:
                h = 2 * torch.sigmoid(self._linear(x.reshape(bs, -1), self.hidden_size)) * h
        return x, h

    def __call__(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        x = self.embedding(x)
        if state:
            x, masking = self._do_feature_masking(x, state[0], self.round_)
            state = (masking, state[1])
        out, (hn, cn) = self.lstm(x, state)
        logit = self.fc(out)

        return logit, (hn.detach(), cn.detach())
