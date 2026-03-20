import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageConditionEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, num_tokens):
        super().__init__()
        mid_channels = max(hidden_size // 4, 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.global_proj = nn.Linear(hidden_size, hidden_size)
        self.num_tokens = num_tokens

    def forward(self, cond_img):
        features = self.stem(cond_img)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.num_tokens).transpose(1, 2)
        cond_global = self.global_proj(tokens.mean(dim=1))
        return cond_global, tokens
