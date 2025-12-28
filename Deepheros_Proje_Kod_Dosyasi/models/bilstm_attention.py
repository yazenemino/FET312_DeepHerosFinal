import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=2, num_classes=8, dropout=0.5, bidirectional=True):
        super(BiLSTMAttention, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(lstm_output_dim, 1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        lstm_out, (hidden, cell) = self.lstm(x)

        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)

        attended = torch.sum(attention_weights * lstm_out, dim=1)

        attended = self.dropout(attended)

        logits = self.fc(attended)

        return logits
