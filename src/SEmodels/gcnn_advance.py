import torch
import torch.nn as nn


class GCNN(nn.Module):
    """
    Gated Convolutional Neural Network with Dropout
    Suitable for medium-scale text classification (e.g. IMDB)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        num_classes=2,
        dropout_emb=0.2,
        dropout_conv=0.3,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(dropout_emb)

        # Gated convolution block 1
        self.conv_a1 = nn.Conv1d(embedding_dim, 64, kernel_size=5, stride=1)
        self.conv_b1 = nn.Conv1d(embedding_dim, 64, kernel_size=5, stride=1)

        # Gated convolution block 2
        self.conv_a2 = nn.Conv1d(64, 64, kernel_size=5, stride=1)
        self.conv_b2 = nn.Conv1d(64, 64, kernel_size=5, stride=1)

        # Dropout after convolution
        self.conv_dropout = nn.Dropout(dropout_conv)

        # Classification head
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x: [B, L]
        """

        # Embedding
        x = self.embedding(x)              # [B, L, E]
        x = self.embedding_dropout(x)
        x = x.transpose(1, 2)              # [B, E, L]

        # Gated Conv Block 1
        a = self.conv_a1(x)
        b = self.conv_b1(x)
        h = a * torch.sigmoid(b)
        h = self.conv_dropout(h)

        # Gated Conv Block 2
        a = self.conv_a2(h)
        b = self.conv_b2(h)
        h = a * torch.sigmoid(b)
        h = self.conv_dropout(h)

        # Global average pooling
        h = h.mean(dim=-1)                 # [B, 64]

        # Output
        logits = self.fc(h)
        return logits
