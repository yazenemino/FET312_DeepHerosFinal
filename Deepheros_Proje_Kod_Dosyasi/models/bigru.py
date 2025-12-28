import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=2, num_classes=8, dropout=0.5, bidirectional=True):
        super(BiGRU, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(gru_output_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        gru_out, hidden = self.gru(x)

        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            x = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            x = hidden[-1]

        x = self.dropout(x)

        logits = self.fc(x)

        return logits
