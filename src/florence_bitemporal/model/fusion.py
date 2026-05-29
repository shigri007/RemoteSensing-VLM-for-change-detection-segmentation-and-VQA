import torch
import torch.nn as nn


class TemporalFusionModule(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.time_emb = nn.Parameter(torch.zeros(2, 1, dim))
        nn.init.trunc_normal_(self.time_emb, std=0.02)

        self.diff_mix = nn.Linear(dim, dim)
        self.diff_gate = nn.Parameter(torch.zeros(1))

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features_t1, features_t2):
        input_dtype = features_t1.dtype
        features_t1 = features_t1.to(torch.float32)
        features_t2 = features_t2.to(torch.float32)

        features_t1_time = features_t1 + self.time_emb[0]
        features_t2_time = features_t2 + self.time_emb[1]

        diff = self.diff_mix(features_t2 - features_t1)
        gate = torch.sigmoid(self.diff_gate)
        features_t1_time = features_t1_time + gate * diff
        features_t2_time = features_t2_time + gate * diff

        combined = torch.cat([features_t1_time, features_t2_time], dim=1)
        attended = self.blocks(combined)
        output = self.norm(attended)

        return output.to(input_dtype)
