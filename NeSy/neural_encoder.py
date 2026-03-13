"""
neural_encoder.py
─────────────────
Neural component of the NeSy model.
Encodes EEG + Watch signals into a shared latent representation.

Input dimensions are determined at runtime from the merged dataset
(passed in via meta["eeg_input_dim"] and meta["watch_input_dim"]).
"""

import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    """Encodes processed EEG features into a latent vector."""

    def __init__(self, input_dim: int, latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        hidden = max(input_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WatchEncoder(nn.Module):
    """Encodes processed Watch (HR, accelerometer) features into a latent vector."""

    def __init__(self, input_dim: int, latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        hidden = max(input_dim, 32)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralEncoder(nn.Module):
    """
    Fuses EEG and Watch encoders into a single latent representation.

    Args:
        eeg_input_dim   : number of processed EEG features  (from meta)
        watch_input_dim : number of processed Watch features (from meta)
        latent_dim      : output latent size (default 64)
        dropout         : dropout probability
    """

    def __init__(
        self,
        eeg_input_dim: int,
        watch_input_dim: int,
        latent_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.eeg_encoder   = EEGEncoder(eeg_input_dim,   latent_dim, dropout)
        self.watch_encoder = WatchEncoder(watch_input_dim, latent_dim, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        eeg_features:   torch.Tensor,   # (batch, eeg_input_dim)
        watch_features: torch.Tensor,   # (batch, watch_input_dim)
    ) -> torch.Tensor:                  # (batch, latent_dim)
        eeg_z   = self.eeg_encoder(eeg_features)
        watch_z = self.watch_encoder(watch_features)
        return self.fusion(torch.cat([eeg_z, watch_z], dim=-1))