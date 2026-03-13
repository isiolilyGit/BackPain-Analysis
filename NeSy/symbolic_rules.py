"""
symbolic_rules.py
─────────────────
Symbolic component of the NeSy model.

Rules are grounded in actual survey columns from survey_answers_en.xlsx.
All thresholds are expressed in the 0-1 normalised (post-StandardScaler) space.
Each rule returns a soft score in [0,1] per class (low / medium / high).

Symbolic rule ↔ survey column mapping
───────────────────────────────────────
rule_pain_severity   ← "How severe is your pain now?"            (Likert 1-5)
rule_daily_impact    ← "How much does pain affect daily activities?" (Likert 1-5)
rule_unbearable      ← "How unbearable is your pain right now?"  (Likert 1-5)
rule_pain_regularity ← "What is the regularity of your back pain?" (ordinal 1-3)
rule_chronic         ← "Do you have any chronic pain conditions?" (0/1/2)
rule_composite       ← severity x daily-impact combined signal
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class RuleOutput:
    scores: torch.Tensor   # (batch, 3)  — scores for [low, medium, high]
    name:   str


class LearnableRuleWeights(nn.Module):
    """One sigmoid-bounded scalar weight per rule, initialised to 1.0."""
    def __init__(self, n_rules: int):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.ones(n_rules))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_weights)


# ── Individual rules ──────────────────────────────────────────────────────────

def rule_pain_severity(survey: torch.Tensor, col: int) -> RuleOutput:
    """
    "How severe is your pain now?" (encoded 1-5, scaled to ~0-1).
    High value → high pain intensity.
    """
    s = torch.sigmoid(survey[:, col]).unsqueeze(1)
    low    = 1 - s
    medium = 1 - (s - 0.5).abs() * 2
    high   = s
    return RuleOutput(torch.cat([low, medium.clamp(0,1), high], dim=1), "Pain severity rule")


def rule_daily_impact(survey: torch.Tensor, col: int) -> RuleOutput:
    """
    "How much does pain currently affect your daily activities?" (Likert 1-5).
    Greater daily disruption → higher pain class.
    """
    s = torch.sigmoid(survey[:, col]).unsqueeze(1)
    low    = 1 - s
    medium = 1 - (s - 0.5).abs() * 2
    high   = s
    return RuleOutput(torch.cat([low, medium.clamp(0,1), high], dim=1), "Daily impact rule")


def rule_unbearable(survey: torch.Tensor, col: int) -> RuleOutput:
    """
    "How unbearable is your pain right now?" (Likert 1–5).
    Used as an autonomic / distress proxy (mirrors HRV inversely).
    """
    s = torch.sigmoid(survey[:, col]).unsqueeze(1)
    low    = 1 - s
    medium = 1 - (s - 0.5).abs() * 2
    high   = s
    return RuleOutput(torch.cat([low, medium.clamp(0,1), high], dim=1), "Unbearable/autonomic rule")


def rule_pain_regularity(survey: torch.Tensor, col: int) -> RuleOutput:
    """
    "What is the regularity of your back pain?" (1=Rarely, 2=Moderately frequent, 3=Frequently).
    Frequent pain → more likely to be chronic/high.
    """
    s = torch.sigmoid(survey[:, col]).unsqueeze(1)
    low    = 1 - s
    medium = 1 - (s - 0.5).abs() * 2
    high   = s
    return RuleOutput(torch.cat([low, medium.clamp(0,1), high], dim=1), "Pain regularity rule")


def rule_chronic_condition(survey: torch.Tensor, col: int) -> RuleOutput:
    """
    "Do you have any chronic pain conditions?" (0=No, 1=Unknown, 2=Yes).
    Chronic condition → shifts probability toward medium/high.
    """
    s = torch.sigmoid(survey[:, col]).unsqueeze(1)
    low    = 1 - s
    medium = s * 0.6
    high   = s * 0.4
    return RuleOutput(torch.cat([low, medium, high], dim=1), "Chronic condition rule")


def rule_composite_severity_impact(
    survey: torch.Tensor, severity_col: int, impact_col: int
) -> RuleOutput:
    """
    Composite: high severity AND high daily impact → strong high-pain signal.
    Low severity AND low impact → strong low-pain signal.
    """
    sev    = torch.sigmoid(survey[:, severity_col])
    impact = torch.sigmoid(survey[:, impact_col])
    high   = (sev * impact).unsqueeze(1)
    low    = ((1 - sev) * (1 - impact)).unsqueeze(1)
    medium = (1 - high - low).clamp(0, 1)
    return RuleOutput(torch.cat([low, medium, high], dim=1), "Composite severity×impact rule")


# ── Rule Engine ───────────────────────────────────────────────────────────────

class SymbolicRuleEngine(nn.Module):
    """
    Applies all 6 clinical rules and combines via learnable weights.

    Column indices within the (scaled) survey tensor:
      vas_col      — "How severe is your pain now?"
      movement_col — "How much does pain affect daily activities?"
      hrv_col      — "How unbearable is your pain right now?"
      regularity_col — "What is the regularity of your back pain?"
      chronic_col  — "Do you have any chronic pain conditions?"
    """

    def __init__(
        self,
        vas_col:        int = 0,
        movement_col:   int = 2,
        hrv_col:        int = 3,
        regularity_col: int = 4,
        chronic_col:    int = 5,
    ):
        super().__init__()
        self.vas_col        = vas_col
        self.movement_col   = movement_col
        self.hrv_col        = hrv_col
        self.regularity_col = regularity_col
        self.chronic_col    = chronic_col
        self.n_rules        = 6
        self.rule_weights   = LearnableRuleWeights(self.n_rules)

    def forward(self, survey: torch.Tensor, latent: torch.Tensor):
        rules = [
            rule_pain_severity(survey,         self.vas_col),
            rule_daily_impact(survey,          self.movement_col),
            rule_unbearable(survey,            self.hrv_col),
            rule_pain_regularity(survey,       self.regularity_col),
            rule_chronic_condition(survey,     self.chronic_col),
            rule_composite_severity_impact(survey, self.vas_col, self.movement_col),
        ]
        weights = self.rule_weights()                              # (n_rules,)
        stacked = torch.stack([r.scores for r in rules], dim=0)   # (n_rules, batch, 3)
        logits  = (stacked * weights[:, None, None]).sum(dim=0)   # (batch, 3)
        return logits, rules

    def explain(
        self,
        survey: torch.Tensor,
        latent: torch.Tensor,
        class_names: List[str] = ["low", "medium", "high"],
    ) -> List[Dict]:
        logits, rule_outputs = self.forward(survey, latent)
        weights  = self.rule_weights().detach()
        preds    = logits.argmax(dim=1)
        results  = []
        for i in range(survey.shape[0]):
            results.append({
                "sample_idx":      i,
                "predicted_class": class_names[preds[i].item()],
                "rule_contributions": [
                    {
                        "rule":             r.name,
                        "rule_prediction":  class_names[r.scores[i].argmax().item()],
                        "weight":           round(weights[j].item(), 3),
                        "scores": {
                            name: round(r.scores[i][k].item(), 3)
                            for k, name in enumerate(class_names)
                        },
                    }
                    for j, r in enumerate(rule_outputs)
                ],
            })
        return results