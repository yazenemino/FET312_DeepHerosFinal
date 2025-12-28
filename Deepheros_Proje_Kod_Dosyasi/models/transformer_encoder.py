import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4,
                 hidden_dim=1024, num_classes=8, max_length=512, dropout=0.1):
        super(TransformerEncoderClassifier, self).__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_scale = math.sqrt(embed_dim)

        self.pos_encoder = PositionalEncoding(embed_dim, max_length, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x) * self.embed_scale

        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        if mask is None:
            mask = (x.sum(dim=-1) != 0)

        attention_mask = mask.float()
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)

        x = self.transformer_encoder(x, src_key_padding_mask=(mask == 0))

        mask_expanded = mask.unsqueeze(-1).expand_as(x).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        x = self.dropout(x)
        logits = self.fc(x)

        return logits
