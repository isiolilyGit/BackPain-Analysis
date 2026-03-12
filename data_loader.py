"""
data_loader.py

Loads, merges, and prepares the PhysioPain dataset for the NeSy model.

Survey labels and features are derived directly from survey_answers_en.xlsx.
Back pain intensity label: "What is the severity of your back pain?" (1-5 scale)
  → 1-2 : low    (class 0)
  → 3   : medium  (class 1)
  → 4-5 : high    (class 2)

Only subjects with a back pain severity score are included in training.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Dict, List

# Data Paths 

BASE_PATH   = Path("Data/PhysioPain Dataset/PhysioPain Dataset")
SURVEY_FILE = BASE_PATH / "SURVEY DATA/survey_answers_en.xlsx"

PATHS = {
    "eeg":   BASE_PATH / "EEG DATA/PROCESSED EEG DATA",
    "watch": BASE_PATH / "WATCH DATA/PROCESSED WATCH DATA",
}

# Column definitions (exact names from survey_answers_en.xlsx) 

SUBJECT_ID_COL = "id"    # e.g. "S006", "S013", ...

# Primary label: back pain severity (1-5 numeric)
BACK_PAIN_SEVERITY_COL = (
    "What is the severity of your back pain? "
    "(You can give a rating from 1 to 5, with 1 representing "
    "the mildest pain and 5 representing the most severe pain.)"
)

# Pain type column — for reference / optional filtering
PAIN_TYPE_COL        = "What is your pain type?"
BACK_PAIN_TYPE_VALUE = "Back/neck/waist pain"

# ── Likert severity features (text → 1–5 integer encoding) ───────────────────

LIKERT_SEVERITY_COLS = [
    "Rate the severity of your pain (Likert scale) [How severe is your pain now?]",
    "Rate the severity of your pain (Likert scale) [How bothersome is your pain right now?]",
    "Rate the severity of your pain (Likert scale) [How much does your pain currently affect your daily activities?]",
    "Rate the severity of your pain (Likert scale) [How unbearable is your pain right now?]",
    "Rate the severity of your pain (Likert scale) [How widespread is your pain now?]",
]

LIKERT_SEVERITY_MAP = {
    "Not severe at all": 1,
    "Mild":              2,
    "Moderate":          3,
    "Severe":            4,
    "Very severe":       5,
}

# Symbolic rule column short aliases → index resolved at runtime in prepare_data()
#   vas_col      = LIKERT_SEVERITY_COLS[0]  "How severe is your pain now?"
#   movement_col = LIKERT_SEVERITY_COLS[2]  "How much does pain affect daily activities?"
#   hrv_col      = LIKERT_SEVERITY_COLS[3]  "How unbearable is pain?" (autonomic proxy)

# ── Additional numeric survey features ───────────────────────────────────────

NUMERIC_SURVEY_COLS = [
    "Age",
    "How many hours do you sleep per day on average?",
    "How many hours did you sleep before this test?",
]

# ── Label map: severity score → class index ───────────────────────────────────

LABEL_MAP   = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
CLASS_NAMES = ["low", "medium", "high"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csvs(folder: Path) -> pd.DataFrame:
    dfs = []
    for f in sorted(folder.rglob("*.csv")):
        df = pd.read_csv(f)
        print(f"  ✅ {f.name} → {df.shape}")
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSVs found in {folder}")
    return pd.concat(dfs, ignore_index=True)


def aggregate_by_subject(df: pd.DataFrame, subject_col: str) -> pd.DataFrame:
    """Average all numeric features per subject."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if subject_col in numeric:
        numeric.remove(subject_col)
    return df.groupby(subject_col)[numeric].mean().reset_index()


def _parse_hours(val) -> float:
    """Convert sleep-hour strings like '6-8 hours' to numeric midpoint."""
    if pd.isna(val):
        return np.nan
    val = str(val).replace(" hours", "").strip()
    if "-" in val:
        parts = val.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan


def load_survey(survey_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and preprocess survey_answers_en.xlsx.

    Returns:
        survey_df       — cleaned DataFrame with subject ID, features, and
                          'pain_label' column (0=low, 1=medium, 2=high)
        survey_feat_cols — ordered list of feature column names used as model input
    """
    df = pd.read_excel(survey_path)
    print(f"  ✅ Survey loaded → {df.shape}  ({df[SUBJECT_ID_COL].nunique()} subjects)")

    # ── Filter: only subjects with a back pain severity rating ───────────────
    df = df[df[BACK_PAIN_SEVERITY_COL].notna()].copy()
    print(f"  📌 Subjects with back pain severity score: {len(df)}")

    # ── Label ─────────────────────────────────────────────────────────────────
    df["pain_label"] = df[BACK_PAIN_SEVERITY_COL].astype(int).map(LABEL_MAP)
    df = df[df["pain_label"].notna()].copy()
    df["pain_label"] = df["pain_label"].astype(int)

    print(f"\n🎯 Label distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"   {name:>8} (class {i}): {(df['pain_label'] == i).sum()} subjects")

    # ── Encode Likert severity columns → integers 1–5 ─────────────────────────
    for col in LIKERT_SEVERITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(LIKERT_SEVERITY_MAP)

    # ── Parse sleep-hour range strings → numeric ──────────────────────────────
    for col in [
        "How many hours do you sleep per day on average?",
        "How many hours did you sleep before this test?",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_hours)

    # ── Final feature column list ─────────────────────────────────────────────
    survey_feat_cols = [
        c for c in LIKERT_SEVERITY_COLS + NUMERIC_SURVEY_COLS
        if c in df.columns and c != SUBJECT_ID_COL
    ]

    return df, survey_feat_cols


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class PainDataset(Dataset):
    def __init__(
        self,
        eeg:    np.ndarray,
        watch:  np.ndarray,
        survey: np.ndarray,
        labels: np.ndarray,
    ):
        self.eeg    = torch.tensor(eeg,    dtype=torch.float32)
        self.watch  = torch.tensor(watch,  dtype=torch.float32)
        self.survey = torch.tensor(survey, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.watch[idx], self.survey[idx], self.labels[idx]


# ── Main preparation function ─────────────────────────────────────────────────

def prepare_data(
    test_size:    float = 0.2,
    batch_size:   int   = 16,
    random_state: int   = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Full pipeline: load → merge → scale → split → DataLoaders.

    Returns:
        train_loader, test_loader, meta dict with dimensions and rule indices.
    """

    # ── 1. Survey ─────────────────────────────────────────────────────────────
    print("📂 Loading Survey...")
    survey_df, survey_feat_cols = load_survey(SURVEY_FILE)

    # ── 2. EEG & Watch ────────────────────────────────────────────────────────
    print("\n📂 Loading EEG (Processed)...")
    eeg_raw = load_csvs(PATHS["eeg"])

    print("\n📂 Loading Watch (Processed)...")
    watch_raw = load_csvs(PATHS["watch"])

    # Normalise subject ID column in EEG/Watch CSVs to match survey "id"
    for df_raw, label in [(eeg_raw, "EEG"), (watch_raw, "Watch")]:

    # Remove accidental spaces in column names
        df_raw.columns = df_raw.columns.str.strip()

        candidates = [
            c for c in df_raw.columns
            if c.lower() in (
                "id",
                "subject_id",
                "participant_id",
                "subject",
                "person_id"
            )
        ]

        if not candidates:
            raise ValueError(
                f"{label} CSV has no recognisable subject ID column. "
                f"Columns found: {df_raw.columns.tolist()}"
            )

    # Rename to standard ID used in the project
        if candidates[0] != SUBJECT_ID_COL:
            df_raw.rename(columns={candidates[0]: SUBJECT_ID_COL}, inplace=True)

    eeg_agg   = aggregate_by_subject(eeg_raw,   SUBJECT_ID_COL)
    watch_agg = aggregate_by_subject(watch_raw, SUBJECT_ID_COL)

    eeg_cols   = [c for c in eeg_agg.columns   if c != SUBJECT_ID_COL]
    watch_cols = [c for c in watch_agg.columns if c != SUBJECT_ID_COL]

    # ── 3. Merge on subject ID ────────────────────────────────────────────────
    survey_slim = survey_df[[SUBJECT_ID_COL, "pain_label"] + survey_feat_cols]

    merged = (
        survey_slim
        .merge(eeg_agg,   on=SUBJECT_ID_COL, how="inner")
        .merge(watch_agg, on=SUBJECT_ID_COL, how="inner", suffixes=("", "_watch"))
    )

    if len(merged) == 0:
        raise ValueError(
            "Merge produced 0 rows — subject IDs do not match between "
            "the survey file and EEG/Watch CSVs.\n"
            f"  Survey IDs (sample): {survey_df[SUBJECT_ID_COL].head(5).tolist()}\n"
            f"  EEG IDs    (sample): {eeg_agg[SUBJECT_ID_COL].head(5).tolist()}"
        )

    print(f"\n📦 Merged dataset: {merged.shape} | Subjects: {merged[SUBJECT_ID_COL].nunique()}")

    # Re-resolve watch columns after merge (suffixes may have shifted names)
    watch_cols_merged = [c for c in merged.columns if c in watch_cols or c.endswith("_watch")]

    # ── 4. Feature matrices ───────────────────────────────────────────────────
    X_eeg    = merged[eeg_cols].fillna(0).values.astype(np.float32)
    X_watch  = merged[watch_cols_merged].fillna(0).values.astype(np.float32)
    X_survey = merged[survey_feat_cols].fillna(
        merged[survey_feat_cols].median()
    ).values.astype(np.float32)
    labels   = merged["pain_label"].values.astype(np.int64)

    # ── 5. Symbolic rule column indices (within survey tensor) ────────────────
    def survey_idx(col: str) -> int:
        return survey_feat_cols.index(col) if col in survey_feat_cols else 0

    vas_col      = survey_idx(LIKERT_SEVERITY_COLS[0])   # "How severe is pain now?"
    movement_col = survey_idx(LIKERT_SEVERITY_COLS[2])   # "Affects daily activities?"
    hrv_col      = survey_idx(LIKERT_SEVERITY_COLS[3])   # "How unbearable?" (autonomic proxy)

    # ── 6. Train / test split ─────────────────────────────────────────────────
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    # ── 7. Scale ──────────────────────────────────────────────────────────────
    eeg_scaler    = StandardScaler()
    watch_scaler  = StandardScaler()
    survey_scaler = StandardScaler()

    X_eeg_train    = eeg_scaler.fit_transform(X_eeg[train_idx])
    X_eeg_test     = eeg_scaler.transform(X_eeg[test_idx])
    X_watch_train  = watch_scaler.fit_transform(X_watch[train_idx])
    X_watch_test   = watch_scaler.transform(X_watch[test_idx])
    X_survey_train = survey_scaler.fit_transform(X_survey[train_idx])
    X_survey_test  = survey_scaler.transform(X_survey[test_idx])
    y_train        = labels[train_idx]
    y_test         = labels[test_idx]

    print(f"\n Feature dimensions:")
    print(f"   EEG features:    {X_eeg_train.shape[1]}")
    print(f"   Watch features:  {X_watch_train.shape[1]}")
    print(f"   Survey features: {X_survey_train.shape[1]}")
    print(f"   Train:           {len(y_train)} samples")
    print(f"   Test:            {len(y_test)} samples")

    # ── 8. DataLoaders ────────────────────────────────────────────────────────
    train_ds = PainDataset(X_eeg_train, X_watch_train, X_survey_train, y_train)
    test_ds  = PainDataset(X_eeg_test,  X_watch_test,  X_survey_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    meta = {
        "eeg_input_dim":    X_eeg_train.shape[1],
        "watch_input_dim":  X_watch_train.shape[1],
        "survey_input_dim": X_survey_train.shape[1],
        "hrv_col":          hrv_col,
        "vas_col":          vas_col,
        "movement_col":     movement_col,
        "eeg_cols":         eeg_cols,
        "watch_cols":       watch_cols_merged,
        "survey_cols":      survey_feat_cols,
        "class_names":      CLASS_NAMES,
        "scalers":          (eeg_scaler, watch_scaler, survey_scaler),
        "label_map":        LABEL_MAP,
        "n_subjects":       len(merged),
    }

    return train_loader, test_loader, meta


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, test_loader, meta = prepare_data()

    print("\n✅ Sanity check — first batch:")
    eeg, watch, survey, labels = next(iter(train_loader))
    print(f"   EEG tensor:    {eeg.shape}")
    print(f"   Watch tensor:  {watch.shape}")
    print(f"   Survey tensor: {survey.shape}")
    print(f"   Labels:        {labels.tolist()}")

    print(f"\n Symbolic rule indices in survey tensor:")
    print(f"   vas_col      = {meta['vas_col']}  → {meta['survey_cols'][meta['vas_col']][:60]}")
    print(f"   movement_col = {meta['movement_col']}  → {meta['survey_cols'][meta['movement_col']][:60]}")
    print(f"   hrv_col      = {meta['hrv_col']}  → {meta['survey_cols'][meta['hrv_col']][:60]}")