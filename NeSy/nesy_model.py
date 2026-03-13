"""
nesy_model.py
─────────────
Full Neural-Symbolic (NeSy) model for back pain intensity classification.

Architecture:
  1. NeuralEncoder      — latent representation from EEG + Watch signals
  2. SymbolicRuleEngine — 6 rules grounded in real survey columns
  3. NeSyClassifier     — learned fusion of neural + symbolic pathways

Symbolic rules use these survey columns (indices resolved in data_loader):
  vas_col      → "How severe is your pain now?"
  movement_col → "How much does pain affect daily activities?"
  hrv_col      → "How unbearable is your pain right now?"
  regularity_col is derived from vas_col + 1 offset (see __init__)
  chronic_col  is derived from vas_col + 2 offset (see __init__)

Loss:
  L = w_final·CE(final) + w_neural·CE(neural) + w_symbolic·CE(symbolic)
      + λ·||rule_weights||²
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from NeSy.neural_encoder import NeuralEncoder # type: ignore
from NeSy.symbolic_rules import SymbolicRuleEngine


class NeSyPainClassifier(nn.Module):
    """
    Neural-Symbolic model for pain intensity classification (low / medium / high).

    All *_input_dim and *_col arguments should come directly from
    the meta dict returned by data_loader.prepare_data().
    """

    def __init__(
        self,
        eeg_input_dim:    int,
        watch_input_dim:  int,
        survey_input_dim: int,
        latent_dim:       int   = 64,
        n_classes:        int   = 3,
        vas_col:          int   = 0,
        movement_col:     int   = 2,
        hrv_col:          int   = 3,
        regularity_col:   int   = 4,
        chronic_col:      int   = 5,
        dropout:          float = 0.3,
    ):
        super().__init__()
        self.n_classes = n_classes

        # ── Neural pathway ─ latent representation + neural head ───────────────────────────────
        self.neural_encoder = NeuralEncoder(
            eeg_input_dim, watch_input_dim, latent_dim, dropout
        )
        self.neural_head = nn.Linear(latent_dim, n_classes)

        # ── Survey projection → fed into symbolic engine ──────────────────────
        self.survey_proj = nn.Sequential(
            nn.Linear(survey_input_dim, survey_input_dim),
            nn.ReLU(),
        )

        # ── Symbolic pathway ──────────────────────────────────────────────────
        self.symbolic_engine = SymbolicRuleEngine(
            vas_col=vas_col,
            movement_col=movement_col,
            hrv_col=hrv_col,
            regularity_col=regularity_col,
            chronic_col=chronic_col,
        )

        # ── Fusion: neural logits + symbolic logits → final prediction ────────
        self.fusion = nn.Sequential(
            nn.Linear(n_classes * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

        # α: how much to trust neural (1.0) vs symbolic (0.0) — learned
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        eeg:    torch.Tensor,    # (batch, eeg_input_dim)
        watch:  torch.Tensor,    # (batch, watch_input_dim)
        survey: torch.Tensor,    # (batch, survey_input_dim)
    ) -> Dict[str, torch.Tensor]:

        # Neural
        latent        = self.neural_encoder(eeg, watch)         # (batch, latent_dim)
        neural_logits = self.neural_head(latent)                # (batch, n_classes)

        # Symbolic (projected survey + latent context)
        survey_proj   = self.survey_proj(survey)
        sym_logits, rule_outputs = self.symbolic_engine(survey_proj, latent)

        # Fusion
        alpha         = torch.sigmoid(self.alpha)
        fused_logits  = self.fusion(
            torch.cat([neural_logits, sym_logits], dim=-1)
        )
        final_logits  = alpha * fused_logits + (1 - alpha) * sym_logits

        return {
            "final_logits":    final_logits,
            "neural_logits":   neural_logits,
            "symbolic_logits": sym_logits,
            "fused_logits":    fused_logits,
            "latent":          latent,
            "alpha":           alpha,
            "rule_outputs":    rule_outputs,
        }

    def explain(
        self,
        eeg:    torch.Tensor,
        watch:  torch.Tensor,
        survey: torch.Tensor,
        class_names: List[str] = ["low", "medium", "high"],
    ) -> List[Dict]:
        """Human-readable symbolic explanation for each sample in the batch."""
        with torch.no_grad():
            out         = self.forward(eeg, watch, survey)
            survey_proj = self.survey_proj(survey)
            return self.symbolic_engine.explain(
                survey_proj, out["latent"], class_names
            )


# ── Loss ──────────────────────────────────────────────────────────────────────

class NeSyLoss(nn.Module):
    """
    Multi-pathway loss that trains both the neural and symbolic pathways jointly.

      L = w_final·CE(final) + w_neural·CE(neural) + w_symbolic·CE(symbolic)
          + λ_rule·||rule_weights||²
    """

    def __init__(
        self,
        w_final:     float = 1.0,
        w_neural:    float = 0.4,
        w_symbolic:  float = 0.4,
        lambda_rule: float = 0.01,
    ):
        super().__init__()
        self.w_final    = w_final
        self.w_neural   = w_neural
        self.w_symbolic = w_symbolic
        self.lambda_rule = lambda_rule
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs:      Dict[str, torch.Tensor],
        targets:      torch.Tensor,
        rule_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        l_final    = self.ce(outputs["final_logits"],    targets)
        l_neural   = self.ce(outputs["neural_logits"],   targets)
        l_symbolic = self.ce(outputs["symbolic_logits"], targets)
        l_reg      = rule_weights.pow(2).mean()

        total = (
            self.w_final    * l_final
            + self.w_neural   * l_neural
            + self.w_symbolic * l_symbolic
            + self.lambda_rule * l_reg
        )

        return total, {
            "loss_total":    total.item(),
            "loss_final":    l_final.item(),
            "loss_neural":   l_neural.item(),
            "loss_symbolic": l_symbolic.item(),
            "loss_rule_reg": l_reg.item(),
        }