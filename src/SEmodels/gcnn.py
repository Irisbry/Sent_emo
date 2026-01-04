import torch
import torch.nn as nn


class GCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv_a1 = nn.Conv1d(embedding_dim, 64, 5, stride=1)
        self.conv_b1 = nn.Conv1d(embedding_dim, 64, 5, stride=1)

        self.conv_a2 = nn.Conv1d(64, 64, 5, stride=1)
        self.conv_b2 = nn.Conv1d(64, 64, 5, stride=1)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)      # [B, L, E]
        x = x.transpose(1, 2)      # [B, E, L]

        a = self.conv_a1(x)
        b = self.conv_b1(x)
        h = a * torch.sigmoid(b)

        a = self.conv_a2(h)
        b = self.conv_b2(h)
        h = a * torch.sigmoid(b)

        h = h.mean(dim=-1)
        return self.fc(h)
