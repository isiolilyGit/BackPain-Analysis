"""
saint_train.py
──────────────
Two-phase training for multimodal SAINT on the PhysioPain dataset.
Uses integrated EEG + Watch + Survey features.

Phase 1 — Contrastive pre-training (no labels)
    Corrupts random features per row and trains the model to distinguish
    original from corrupted samples (SCARF strategy).
    Essential given only 33 subjects.

Phase 2 — Supervised fine-tuning
    Cross-entropy classification. Encoder frozen for first N warm-up epochs,
    then fully unfrozen at a lower learning rate.

Outputs:
    saint_pretrained.pt         — encoder weights after Phase 1
    saint_best.pt               — best checkpoint after Phase 2
    saint_training_log.csv      — per-epoch loss and accuracy
    saint_feature_importance.csv
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from saint_data import prepare_data, CLASS_NAMES
from saint_model import SAINT, ContrastiveLoss, corrupt_batch

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "dim":             32,
    "depth":           3,
    "n_heads":         4,
    "mlp_ratio":       4,
    "dropout":         0.1,
    "mlp_hidden_mult": 4,
    # Pre-training
    "pretrain_epochs":  50,
    "pretrain_lr":      1e-3,
    "corruption_rate":  0.3,
    "contrastive_temp": 0.07,
    # Fine-tuning
    "finetune_epochs":  100,
    "finetune_lr":      5e-4,
    "warmup_epochs":    10,
    "weight_decay":     1e-4,
    "patience":         20,
    # Data
    "batch_size":    16,
    "test_size":     0.2,
    "random_state":  42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Rule names matching SymbolicRuleEngine order (for display only)
MODALITY_GROUPS = {
    "Survey (categorical)": lambda cols: [c for c in cols if not c.startswith(("eeg_", "watch_")) and c != "Age"],
    "EEG (aggregated)":     lambda cols: [c for c in cols if c.startswith("eeg_")],
    "Watch (aggregated)":   lambda cols: [c for c in cols if c.startswith("watch_")],
}


# ── Phase 1: Contrastive pre-training ─────────────────────────────────────────

def pretrain(model, train_loader, config):
    device    = config["device"]
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["pretrain_lr"],
                            weight_decay=config["weight_decay"])
    criterion = ContrastiveLoss(temperature=config["contrastive_temp"])

    print("\n" + "="*55)
    print("Phase 1 — Contrastive Pre-training")
    print("="*55)
    print(f"{'Epoch':>6}  {'Contrastive Loss':>18}")
    print("-"*30)

    model.train()
    for epoch in range(1, config["pretrain_epochs"] + 1):
        epoch_loss = 0.0
        for x_cat, x_cont, _ in train_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            x_cat_c, x_cont_c = corrupt_batch(x_cat, x_cont, config["corruption_rate"])
            loss = criterion(model.encode(x_cat, x_cont),
                             model.encode(x_cat_c, x_cont_c))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {epoch_loss/len(train_loader):>18.4f}")

    torch.save(model.state_dict(), "saint_pretrained.pt")
    print(f"\n✅ Pre-training complete → saint_pretrained.pt")
    return model


# ── Phase 2: Supervised fine-tuning ───────────────────────────────────────────

def finetune(model, train_loader, test_loader, config):
    device    = config["device"]
    criterion = nn.CrossEntropyLoss()

    def set_encoder_grad(req):
        for name, p in model.named_parameters():
            if "head" not in name:
                p.requires_grad = req

    # Start frozen
    set_encoder_grad(False)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["finetune_lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["finetune_epochs"])

    best_val_acc = 0.0
    patience_cnt = 0
    log = []

    print("\n" + "="*70)
    print("Phase 2 — Supervised Fine-tuning  (EEG + Watch + Survey → SAINT)")
    print("="*70)
    print(f"{'Epoch':>6} {'Tr Loss':>9} {'Tr Acc':>8} {'Va Loss':>9} {'Va Acc':>8} {'Status':>10}")
    print("-"*70)

    for epoch in range(1, config["finetune_epochs"] + 1):

        # Unfreeze after warm-up
        if epoch == config["warmup_epochs"] + 1:
            set_encoder_grad(True)
            optimizer = optim.AdamW(model.parameters(),
                                    lr=config["finetune_lr"] * 0.1,
                                    weight_decay=config["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["finetune_epochs"] - epoch)
            print(f"  ── Epoch {epoch}: encoder unfrozen, full model fine-tuning ──")

        status = "frozen" if epoch <= config["warmup_epochs"] else "unfrozen"

        # Train
        model.train()
        tr_loss = tr_c = tr_n = 0
        for x_cat, x_cont, labels in train_loader:
            x_cat, x_cont, labels = x_cat.to(device), x_cont.to(device), labels.to(device)
            logits = model(x_cat, x_cont)
            loss   = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_c += (logits.argmax(1) == labels).sum().item()
            tr_n += labels.size(0); tr_loss += loss.item()
        scheduler.step()

        # Evaluate
        model.eval()
        va_loss = va_c = va_n = 0
        all_p, all_l = [], []
        with torch.no_grad():
            for x_cat, x_cont, labels in test_loader:
                x_cat, x_cont, labels = x_cat.to(device), x_cont.to(device), labels.to(device)
                logits = model(x_cat, x_cont)
                va_loss += criterion(logits, labels).item()
                preds = logits.argmax(1)
                va_c += (preds == labels).sum().item()
                va_n += labels.size(0)
                all_p.extend(preds.cpu().numpy())
                all_l.extend(labels.cpu().numpy())

        tr_acc = tr_c / tr_n; va_acc = va_c / va_n
        log.append(dict(epoch=epoch,
                        train_loss=tr_loss/len(train_loader), train_acc=tr_acc,
                        val_loss=va_loss/len(test_loader),    val_acc=va_acc))
        print(f"{epoch:>6} {tr_loss/len(train_loader):>9.4f} {tr_acc:>8.3f} "
              f"{va_loss/len(test_loader):>9.4f} {va_acc:>8.3f} {status:>10}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), "saint_best.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config["patience"]:
                print(f"\n⏹️  Early stopping at epoch {epoch} "
                      f"(best val acc: {best_val_acc:.3f})")
                break

    pd.DataFrame(log).to_csv("saint_training_log.csv", index=False)
    return best_val_acc, all_p, all_l


# ── Feature importance ────────────────────────────────────────────────────────

@torch.no_grad()
def feature_importance(model, test_loader, meta, device, top_n=20):
    """
    Estimate feature importance via cosine similarity of each feature token
    to the CLS token, averaged over the test set.
    """
    model.eval()
    all_feat_cols = meta["cat_cols"] + meta["cont_cols"]
    scores = torch.zeros(len(all_feat_cols))

    for x_cat, x_cont, _ in test_loader:
        x_cat, x_cont = x_cat.to(device), x_cont.to(device)
        emb  = model.embed(x_cat, x_cont)          # (B, N+1, dim)
        emb  = model.blocks(emb)
        cls  = emb[:, 0]                            # (B, dim)
        feat = emb[:, 1:]                           # (B, N, dim)
        sim  = torch.nn.functional.cosine_similarity(
            feat, cls.unsqueeze(1).expand_as(feat), dim=-1)   # (B, N)
        scores += sim.mean(0).cpu()

    scores /= len(test_loader)
    imp_df = pd.DataFrame({"feature": all_feat_cols, "importance": scores.numpy()})
    imp_df["modality"] = imp_df["feature"].apply(
        lambda c: "EEG" if c.startswith("eeg_")
        else ("Watch" if c.startswith("watch_") else "Survey"))
    imp_df = imp_df.sort_values("importance", ascending=False)

    print(f"\n🏆 Top {top_n} Features by Modality:")
    for modality in ["EEG", "Watch", "Survey"]:
        sub = imp_df[imp_df["modality"] == modality].head(5)
        print(f"\n  [{modality}]")
        for _, row in sub.iterrows():
            bar = "█" * int(row["importance"] * 25)
            print(f"    {row['importance']:.3f}  {bar}  {row['feature'][:65]}")

    imp_df.to_csv("saint_feature_importance.csv", index=False)
    print("\n💾 Saved → saint_feature_importance.csv")
    return imp_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = CONFIG["device"]
    print(f"\n🖥️  Device: {device}")

    train_loader, test_loader, meta = prepare_data(
        test_size=CONFIG["test_size"],
        batch_size=CONFIG["batch_size"],
        random_state=CONFIG["random_state"],
    )

    model = SAINT(
        cat_dims        = meta["cat_dims"],
        n_cont          = meta["n_cont"],
        dim             = CONFIG["dim"],
        depth           = CONFIG["depth"],
        n_heads         = CONFIG["n_heads"],
        mlp_ratio       = CONFIG["mlp_ratio"],
        dropout         = CONFIG["dropout"],
        mlp_hidden_mult = CONFIG["mlp_hidden_mult"],
        n_classes       = meta["n_classes"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 SAINT — {n_params:,} parameters")
    print(f"   Categorical features : {meta['n_cat']}  (survey ordinal/binary)")
    print(f"   Continuous features  : {meta['n_cont']}  "
          f"(Age + {len(meta['eeg_feat_cols'])} EEG agg + {len(meta['watch_feat_cols'])} Watch agg)")
    print(f"   Embedding dim        : {CONFIG['dim']}")
    print(f"   Transformer depth    : {CONFIG['depth']}  blocks")
    print(f"   Subjects             : {len(meta['subject_ids'])}  "
          f"({meta['subject_ids'][:5]} ...)")

    # Phase 1
    model = pretrain(model, train_loader, CONFIG)

    # Phase 2
    best_acc, preds, labels = finetune(model, train_loader, test_loader, CONFIG)
    print(f"\n✅ Best validation accuracy: {best_acc:.3f}")

    # Final evaluation on best checkpoint
    model.load_state_dict(torch.load("saint_best.pt", map_location=device))
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for x_cat, x_cont, lbl in test_loader:
            logits = model(x_cat.to(device), x_cont.to(device))
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_l.extend(lbl.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_l, all_p, target_names=CLASS_NAMES,
                                 zero_division=0))
    print("📊 Confusion Matrix  (rows=true, cols=predicted):")
    cm = confusion_matrix(all_l, all_p)
    header = "         " + "  ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>6}   " + "  ".join(f"{v:>8}" for v in row))

    # Feature importance
    feature_importance(model, test_loader, meta, device)


if __name__ == "__main__":
    main()