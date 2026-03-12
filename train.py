"""
train.py

Training loop for the NeSy back pain intensity classifier.

Usage:
    python train.py

Outputs:
    best_nesy_model.pt  — best checkpoint (by validation accuracy)
    training_log.csv    — per-epoch metrics
"""

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import prepare_data, CLASS_NAMES
from nesy_model import NeSyPainClassifier, NeSyLoss

# Config 

CONFIG = {
    "latent_dim":   64,
    "dropout":      0.3,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "epochs":       80,
    "batch_size":   16,
    "test_size":    0.2,
    "patience":     15,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

# Actual rule names — must match order in SymbolicRuleEngine.forward()
RULE_NAMES = [
    "Pain severity           [How severe is pain now?]",
    "Daily impact            [Affects daily activities?]",
    "Unbearable/autonomic    [How unbearable?]",
    "Pain regularity         [Rarely→Frequently]",
    "Chronic condition       [Has chronic pain?]",
    "Composite severity×impact",
]


# Utilities 

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for eeg, watch, survey, labels in loader:
        eeg, watch, survey, labels = (
            eeg.to(device), watch.to(device), survey.to(device), labels.to(device))
        optimizer.zero_grad()
        outputs      = model(eeg, watch, survey)
        rule_weights = model.symbolic_engine.rule_weights()
        loss, _      = criterion(outputs, labels, rule_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds    = outputs["final_logits"].argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item()
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for eeg, watch, survey, labels in loader:
        eeg, watch, survey, labels = (
            eeg.to(device), watch.to(device), survey.to(device), labels.to(device))
        outputs      = model(eeg, watch, survey)
        rule_weights = model.symbolic_engine.rule_weights()
        loss, _      = criterion(outputs, labels, rule_weights)
        preds        = outputs["final_logits"].argmax(1)
        correct     += (preds == labels).sum().item()
        total       += labels.size(0)
        total_loss  += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels


#  Main. Training loop, evaluation, and explanations of the NeSy model.

def train():
    device = CONFIG["device"]
    print(f"\n  Device: {device}\n")

    train_loader, test_loader, meta = prepare_data(
        test_size=CONFIG["test_size"],
        batch_size=CONFIG["batch_size"],
    )

    # Resolve regularity and chronic column indices from survey_cols
    survey_cols = meta["survey_cols"]
    def sidx(keyword):
        matches = [i for i, c in enumerate(survey_cols) if keyword.lower() in c.lower()]
        return matches[0] if matches else meta["vas_col"]

    regularity_col = sidx("regularity")
    chronic_col    = sidx("chronic pain")

    model = NeSyPainClassifier(
        eeg_input_dim    = meta["eeg_input_dim"],
        watch_input_dim  = meta["watch_input_dim"],
        survey_input_dim = meta["survey_input_dim"],
        latent_dim       = CONFIG["latent_dim"],
        n_classes        = 3,
        vas_col          = meta["vas_col"],
        movement_col     = meta["movement_col"],
        hrv_col          = meta["hrv_col"],
        regularity_col   = regularity_col,
        chronic_col      = chronic_col,
        dropout          = CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f" Model parameters: {n_params:,}")
    print(f"   Neural-symbolic alpha (initial): {torch.sigmoid(model.alpha).item():.2f}")
    print(f"\n   Symbolic rule column mapping:")
    print(f"   vas_col={meta['vas_col']}         → {survey_cols[meta['vas_col']][:60]}")
    print(f"   movement_col={meta['movement_col']}    → {survey_cols[meta['movement_col']][:60]}")
    print(f"   hrv_col={meta['hrv_col']}         → {survey_cols[meta['hrv_col']][:60]}")
    print(f"   regularity_col={regularity_col}   → {survey_cols[regularity_col][:60]}")
    print(f"   chronic_col={chronic_col}         → {survey_cols[chronic_col][:60]}\n")

    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = NeSyLoss(w_final=1.0, w_neural=0.4, w_symbolic=0.4, lambda_rule=0.01)

    best_val_acc = 0
    patience_cnt = 0
    log = []

    print("=" * 65)
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9} {'α':>6}")
    print("=" * 65)

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        alpha = torch.sigmoid(model.alpha).item()
        log.append(dict(epoch=epoch,
                        train_loss=tr_loss, train_acc=tr_acc,
                        val_loss=va_loss,   val_acc=va_acc,
                        alpha=alpha))
        print(f"{epoch:>6} {tr_loss:>11.4f} {tr_acc:>9.3f} "
              f"{va_loss:>10.4f} {va_acc:>9.3f} {alpha:>6.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), "best_nesy_model.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val acc: {best_val_acc:.3f})")
                break

    pd.DataFrame(log).to_csv("training_log.csv", index=False)
    print(f"\n✅ Training complete.  Best val accuracy: {best_val_acc:.3f}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load("best_nesy_model.pt", map_location=device))
    _, _, preds, labels = evaluate(model, test_loader, criterion, device)

    print("\n📊 Classification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    print("📊 Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(labels, preds))

    # ── Learned rule weights ──────────────────────────────────────────────────
    rule_weights = torch.sigmoid(
        model.symbolic_engine.rule_weights.raw_weights).detach()
    print("\nLearned Rule Weights (higher = model trusted this rule more):")
    for name, w in zip(RULE_NAMES, rule_weights):
        # bar = "█" * int(w.item() * 20)
        print(f"  {name:<50} {w.item():.3f} ")

    print(f"\n🔀 Neural vs Symbolic balance  alpha = {torch.sigmoid(model.alpha).item():.3f}")
    print("   (alpha → 1.0 : fully neural  |  alpha → 0.0 : fully symbolic)")

    #  Per-subject explanations
    print("\n Sample Explanations (first 3 test subjects):")
    eeg_b, watch_b, survey_b, true_b = next(iter(test_loader))
    explanations = model.explain(
        eeg_b[:3].to(device),
        watch_b[:3].to(device),
        survey_b[:3].to(device),
    )
    for exp in explanations:
        true_cls = CLASS_NAMES[true_b[exp["sample_idx"]].item()]
        print(f"\n  Subject {exp['sample_idx']+1}")
        print(f"    True label      : {true_cls.upper()}")
        print(f"    NeSy prediction : {exp['predicted_class'].upper()}")
        print(f"    Rule breakdown  :")
        for r in exp["rule_contributions"]:
            print(f"      [{r['rule']:<50}] "
                  f"w={r['weight']:.2f}  → {r['rule_prediction']}"
                  f"  (low={r['scores']['low']:.2f}, "
                  f"med={r['scores']['medium']:.2f}, "
                  f"high={r['scores']['high']:.2f})")


if __name__ == "__main__":
    train()