"""
saint_data.py
─────────────
Integrates EEG, Watch, and Survey data for SAINT tabular classification.

Pipeline
────────
1. Survey  → encode 70 categorical/continuous features + pain label (low/med/high)
2. EEG     → back_pain_combined_1hz.csv (one row per timestamp per subject)
             aggregate per subject: mean, std, min, max, median of each band
             → 10 bands x 5 stats = 50 EEG features
3. Watch   → one CSV per subject named {ID}_4Hz.csv
             aggregate per subject: mean, std, min, max, median of each signal
             → 6 signals x 5 stats = 30 Watch features
4. Merge   → inner join on subject ID → 33 subjects x ~150 features
5. SAINT split → categorical (survey) vs continuous (EEG + Watch + Age)

Usable subjects: 33 (intersection of survey back-pain cohort + EEG data)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Dict, List

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_PATH   = Path("Data/PhysioPain Dataset/PhysioPain Dataset")
SURVEY_FILE = BASE_PATH / "SURVEY DATA/survey_answers_en.xlsx"
EEG_FILE    = BASE_PATH / "EEG DATA/PROCESSED EEG DATA/back_pain_combined_1hz.csv"
WATCH_DIR   = BASE_PATH / "WATCH DATA/PROCESSED WATCH DATA"

# ── Label ─────────────────────────────────────────────────────────────────────

BACK_PAIN_COL = (
    "What is the severity of your back pain? "
    "(You can give a rating from 1 to 5, with 1 representing "
    "the mildest pain and 5 representing the most severe pain.)"
)
LABEL_MAP   = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
CLASS_NAMES = ["low", "medium", "high"]

# ── EEG columns ───────────────────────────────────────────────────────────────

EEG_SIGNAL_COLS = [
    " Delta", " Theta", " Alpha1", " Alpha2",
    " Beta1", " Beta2", " Gamma1", " Gamma2",
    " Attention", " Meditation",
]
EEG_ID_COL = "person_id"

# ── Watch columns ─────────────────────────────────────────────────────────────

WATCH_SIGNAL_COLS = ["bvp", "eda", "x", "y", "z", "temperature"]
WATCH_ID_COL      = "person_id"

# ── Statistical aggregation functions ────────────────────────────────────────

STAT_FUNCS = {
    "mean":   np.mean,
    "std":    np.std,
    "min":    np.min,
    "max":    np.max,
    "median": np.median,
}


def aggregate_timeseries(
    df: pd.DataFrame,
    id_col: str,
    signal_cols: List[str],
    prefix: str,
) -> pd.DataFrame:
    """
    Aggregate time-series rows per subject into a single-row feature vector.
    For each signal column: computes mean, std, min, max, median.
    Returns a DataFrame with one row per subject.
    """
    records = []
    for subject_id, group in df.groupby(id_col):
        row = {id_col: subject_id}
        for col in signal_cols:
            vals = group[col].dropna().values.astype(np.float64)
            if len(vals) == 0:
                for stat in STAT_FUNCS:
                    row[f"{prefix}_{col.strip()}_{stat}"] = np.nan
            else:
                for stat, fn in STAT_FUNCS.items():
                    row[f"{prefix}_{col.strip()}_{stat}"] = fn(vals)
        records.append(row)
    return pd.DataFrame(records)


# ── Survey encoding (identical to NeSy data_loader) ──────────────────────────

DROP_COLS = {
    "id", BACK_PAIN_COL,
    "What is your pain type?",
    "The district you live in (For weather quality)",
    "What region is your family origin from?",
    "Class",
    "Has your signal data been collected within the scope of the project?",
    "Did you do any work during the test?\nSignal data if received",
    "Do you have any personal concerns or hesitations arising from the use of EEG (brain signal) or similar devices?",
    "Where is your back pain localized? (Ex: neck, upper back, lower back, etc.)",
    "How would you describe your type of back pain? (Continuous pain, muscle spasm, sudden stinging pain, etc.)",
    "How long has your back been hurting?",
    "What do you think is causing your back pain?",
    "How would you best describe the nature of your complaint?",
    "At what times of the day do you feel your pain?",
}
DROP_LOW_VARIANCE = {"Group 8", "Group 18", "Do you have nausea?"}

SLEEP_AVG_MAP = {"Less than 4 hours": 0, "4-6 hours": 1, "6-8 hours": 2, "More than 8 hours": 3}
SLEEP_BEFORE_MAP = {"Did not sleep at all": 0, "3-4 hours": 1, "5-6 hours": 2, "7-8 hours": 3, "9 hours or more": 4}
STRESS_MAP    = {"Low": 0, "Medium": 1, "High": 2}
YESNO_MAP     = {"No": 0, "Yes": 1, "Hayır": 0, "Evet": 1}
GENDER_MAP    = {"Female": 0, "Male": 1}
INCOME_MAP    = {"Low": 0, "Middle": 1, "High": 2}
FAMILY_MAP    = {"Single": 0, "Married": 1, "Divorced": 2}
EDUCATION_MAP = {"High School": 0, "University": 1, "Master": 2}
ALCOHOL_MAP   = {"I never use alcohol.": 0, "I use alcohol occasionally.": 1, "I use alcohol regularly.": 2}
SMOKE_MAP     = {"Never": 0, "Sometimes": 1, "Most of the time": 2, "Yes": 3}
EXERCISE_MAP  = {
    "I do not exercise at all.": 0, "I never exercise regularly.": 0,
    "I exercise sometimes.": 1, "I exercise most of the time.": 2, "I always exercise.": 3,
}
EATING_MAP    = {"I never eat healthy.": 0, "I sometimes eat healthy.": 1, "I mostly eat healthy.": 2, "I always eat healthy.": 3}
WATER_BEFORE_MAP = {"Did not drink at all": 0, "A few sips": 1, "About a glass": 2, "A few glasses": 3, "One liter or more": 4}
WATER_AVG_MAP = {
    "I do not drink water daily": 0, "I drink one glass or less daily": 1,
    "I drink a few glasses daily": 2, "I drink at least half a liter of water daily": 3,
    "I drink one liter or more water daily": 4,
}
CHRONIC_MAP  = {"No": 0, "Do not know": 1, "Yes": 2}
FEELING_MAP  = {
    "Relaxed": 0, "Peaceful": 0, "Happy": 1, "Euphoric": 1,
    "Tired": 2, "Perplexed": 2, "Sad": 3, "Stressed": 3, "Tense": 4, "Anxious": 4,
}
PROFESSION_MAP = {"Student": 0, "Freelancer": 1, "Private Sector": 2, "Worker": 3, "Retired": 4}
LIKERT_MAP = {"Not severe at all": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Very severe": 4}
LIKERT_COLS = [
    "Rate the severity of your pain (Likert scale) [How severe is your pain now?]",
    "Rate the severity of your pain (Likert scale) [How bothersome is your pain right now?]",
    "Rate the severity of your pain (Likert scale) [How much does your pain currently affect your daily activities?]",
    "Rate the severity of your pain (Likert scale) [How unbearable is your pain right now?]",
    "Rate the severity of your pain (Likert scale) [How widespread is your pain now?]",
]
PAIN_DURATION_MAP   = {"Sudden and short-lived (Several minutes to several hours)": 0, "Sudden and medium-lived (Several hours to several days)": 1, "Long-lasting (Several days to several weeks)": 2, "Very long-lasting (More than several weeks)": 3}
PAIN_ONSET_MAP      = {"Recent time (within the last few weeks)": 0, "Medium time (within the last few months)": 1, "Long time (within the last few years)": 2}
PAIN_REGULARITY_MAP = {"Rarely": 0, "Moderately frequent": 1, "Frequently": 2}
PAIN_IMPACT_MAP     = {"Does not affect": 0, "Slightly affects": 1, "Moderately affects": 2, "Greatly affects": 3}
PAIN_SITUATION_MAP  = {"Intermittent": 0, "Constant": 1}
PAIN_START_MAP      = {"Gradually": 0, "Suddenly": 1}
PAIN_TEMPORAL_MAP   = {"Rhythmic, periodic, intermittent": 0, "Short, momentary, transient": 1, "Continuous, constant, steady": 2}
DIZZINESS_MAP       = {"Never": 0, "Sometimes": 1, "Mostly": 2}
FREQUENCY_MAP       = {"Never": 0, "Sometimes": 1, "Almost": 2, "Often": 3, "Always": 4}
PHYSICAL_COLS = [
    "Physical Factor Table [I am asked to do too much work]",
    "Physical Factor Table [I need to work out a lot]",
    "Physical Factor Table [I enjoy my name]",
    "Physical Factor Table [I have enough time to do my name]",
    "Physical Factor Table [I am asked to perform unrealistic tasks]",
    "Physical Factor Table [I do a monotonous job]",
    "Physical Factor Table [I lift heavy loads]",
    "Physical Factor Table [I lift light loads]",
]

def parse_mcgill(val):
    if pd.isna(val): return np.nan
    try:    return int(str(val).split("-")[0].strip()) - 1
    except: return np.nan


def encode_survey(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Encode all survey columns. Returns (encoded_df, survey_feature_col_names)."""
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Gender (Biological)"] = df["Gender (Biological)"].map(GENDER_MAP)
    df["How many hours do you sleep per day on average?"] = df["How many hours do you sleep per day on average?"].map(SLEEP_AVG_MAP)
    df["How many hours did you sleep before this test?"]  = df["How many hours did you sleep before this test?"].map(SLEEP_BEFORE_MAP)
    df["Have you had any problems falling asleep or staying asleep? (Yes No)"] = df["Have you had any problems falling asleep or staying asleep? (Yes No)"].map(YESNO_MAP)
    df["How have you been feeling in general lately?"]    = df["How have you been feeling in general lately?"].map(FEELING_MAP)
    df["What do you think about your stress level in your daily life?"] = df["What do you think about your stress level in your daily life?"].map(STRESS_MAP)
    df["Have you experienced any changes or problems in your social relationships?"] = df["Have you experienced any changes or problems in your social relationships?"].map(YESNO_MAP)
    df["educational background"]  = df["educational background"].map(EDUCATION_MAP)
    df["Which professional group do you belong to?"] = df["Which professional group do you belong to?"].map(PROFESSION_MAP)
    df["Income rate"]             = df["Income rate"].map(INCOME_MAP)
    df["family situation"]        = df["family situation"].map(FAMILY_MAP)
    df["How would you rate yourself based on your frequency of alcohol use?"] = df["How would you rate yourself based on your frequency of alcohol use?"].map(ALCOHOL_MAP)
    df["Have you consumed alcohol before this test?"] = df["Have you consumed alcohol before this test?"].map(YESNO_MAP)
    df["Do you smoke?"]           = df["Do you smoke?"].map(SMOKE_MAP)
    df["Have you smoked before this test?"] = df["Have you smoked before this test?"].map(YESNO_MAP)
    df["How would you rate yourself based on your exercise frequency?"] = df["How would you rate yourself based on your exercise frequency?"].map(EXERCISE_MAP)
    df["How would you evaluate yourself based on your eating habits?"]  = df["How would you evaluate yourself based on your eating habits?"].map(EATING_MAP)
    df["Do you think you drink enough water?"]        = df["Do you think you drink enough water?"].map(YESNO_MAP)
    df["How much water did you drink before this test?"] = df["How much water did you drink before this test?"].map(WATER_BEFORE_MAP)
    df["What is your general water drinking average?"]   = df["What is your general water drinking average?"].map(WATER_AVG_MAP)
    df["Do you have any chronic pain conditions?"]    = df["Do you have any chronic pain conditions?"].map(CHRONIC_MAP)
    df["Are there any chronic diseases or genetic disorders in your family?"] = df["Are there any chronic diseases or genetic disorders in your family?"].map(CHRONIC_MAP)
    df["Are there any medications or supplements you currently use regularly?"] = df["Are there any medications or supplements you currently use regularly?"].map(YESNO_MAP)
    df["Do you approve of video recording? (If your signal data is collected)"] = df["Do you approve of video recording? (If your signal data is collected)"].map(YESNO_MAP)
    for col in LIKERT_COLS:
        df[col] = df[col].map(LIKERT_MAP)
    for grp in [f"Group {i}" for i in range(1, 21)]:
        df[grp] = df[grp].apply(parse_mcgill)
    df["How does your pain change over time? Tick \u200b\u200bthe most appropriate group."] = df["How does your pain change over time? Tick \u200b\u200bthe most appropriate group."].map(PAIN_TEMPORAL_MAP)
    df["Is there any dizziness?"] = df["Is there any dizziness?"].map(DIZZINESS_MAP)
    df["How long does the pain last? (Sudden and short-term or long-term?)"]   = df["How long does the pain last? (Sudden and short-term or long-term?)"].map(PAIN_DURATION_MAP)
    df["Can you please report when your back pain started and when it got worse? (Recent time, long time)"] = df["Can you please report when your back pain started and when it got worse? (Recent time, long time)"].map(PAIN_ONSET_MAP)
    df["Are there any factors that initiate back pain? (Poor posture, lack of exercise, trauma, etc.)"] = df["Are there any factors that initiate back pain? (Poor posture, lack of exercise, trauma, etc.)"].map(YESNO_MAP)
    df["What is the regularity of your back pain?"]          = df["What is the regularity of your back pain?"].map(PAIN_REGULARITY_MAP)
    df["How does back pain affect your daily living activities?"] = df["How does back pain affect your daily living activities?"].map(PAIN_IMPACT_MAP)
    df["Are you doing anything to relieve your back pain? (Massage, exercise, using painkillers, etc.)"] = df["Are you doing anything to relieve your back pain? (Massage, exercise, using painkillers, etc.)"].map(YESNO_MAP)
    df["How did the pain start?"]     = df["How did the pain start?"].map(PAIN_START_MAP)
    df["How is your pain situation?"] = df["How is your pain situation?"].map(PAIN_SITUATION_MAP)
    df["Do you suffer from fever or chills at night?"] = df["Do you suffer from fever or chills at night?"].map(YESNO_MAP)
    for col in PHYSICAL_COLS:
        df[col] = df[col].map(FREQUENCY_MAP)

    all_drop = DROP_COLS | DROP_LOW_VARIANCE
    menstrual_headache = [c for c in df.columns if any(k in c.lower() for k in ["menstrual", "period", "headache", "migraine"])]
    all_drop.update(menstrual_headache)
    df = df.drop(columns=[c for c in all_drop if c in df.columns])

    feat_cols = [c for c in df.columns if c not in {"id", "pain_label"}]
    return df, feat_cols


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_survey(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_excel(path)
    df = df[df[BACK_PAIN_COL].notna()].copy()
    df["pain_label"] = df[BACK_PAIN_COL].astype(int).map(LABEL_MAP).astype(int)
    print(f"  ✅ Survey: {len(df)} back-pain subjects")
    print(f"     Label distribution: {df['pain_label'].value_counts().sort_index().to_dict()}")
    df, feat_cols = encode_survey(df)
    # Impute categoricals with mode, Age with median
    for c in feat_cols:
        if df[c].isna().sum() > 0:
            fill = df[c].median() if c == "Age" else df[c].mode()[0]
            df[c] = df[c].fillna(fill)
    return df[["id", "pain_label"] + feat_cols], feat_cols


def load_eeg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  ✅ EEG raw: {df.shape}  ({df[EEG_ID_COL].nunique()} subjects, "
          f"~{int(df.shape[0]/df[EEG_ID_COL].nunique())} rows/subject)")
    agg = aggregate_timeseries(df, EEG_ID_COL, EEG_SIGNAL_COLS, prefix="eeg")
    print(f"     After aggregation: {agg.shape}  "
          f"({len([c for c in agg.columns if c != EEG_ID_COL])} features/subject)")
    agg.rename(columns={EEG_ID_COL: "id"}, inplace=True)
    return agg


def load_watch(watch_dir: Path) -> pd.DataFrame:
    """Load all per-subject watch CSVs, aggregate, and return one row per subject."""
    files = sorted(watch_dir.glob("*_4Hz.csv"))
    if not files:
        # Fallback: try any CSV in the directory
        files = sorted(watch_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No watch CSVs found in {watch_dir}")

    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  ✅ Watch raw: {combined.shape}  ({combined[WATCH_ID_COL].nunique()} subjects, "
          f"~{int(combined.shape[0]/combined[WATCH_ID_COL].nunique())} rows/subject)")

    agg = aggregate_timeseries(combined, WATCH_ID_COL, WATCH_SIGNAL_COLS, prefix="watch")
    print(f"     After aggregation: {agg.shape}  "
          f"({len([c for c in agg.columns if c != WATCH_ID_COL])} features/subject)")
    agg.rename(columns={WATCH_ID_COL: "id"}, inplace=True)
    return agg


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SAINTDataset(Dataset):
    """
    Returns (x_cat, x_cont, label) per subject.
      x_cat  : (n_cat,)  int64   — encoded survey categorical features
      x_cont : (n_cont,) float32 — standardised EEG + Watch + continuous survey features
      label  : int64
    """
    def __init__(self, x_cat, x_cont, labels):
        self.x_cat  = torch.tensor(x_cat,  dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_cont[idx], self.labels[idx]


# ── prepare_data ──────────────────────────────────────────────────────────────

def prepare_data(
    test_size:    float = 0.2,
    batch_size:   int   = 16,
    random_state: int   = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Full pipeline: load → aggregate → merge → split → DataLoaders.

    SAINT feature split:
      Categorical : survey ordinal/binary columns (passed to embedding tables)
      Continuous  : EEG aggregated stats + Watch aggregated stats + Age
                    (passed through linear projections, StandardScaler applied)
    """

    print("📂 Loading Survey...")
    survey_df, survey_feat_cols = load_survey(SURVEY_FILE)

    print("\n📂 Loading EEG...")
    eeg_df = load_eeg(EEG_FILE)

    print("\n📂 Loading Watch...")
    watch_df = load_watch(WATCH_DIR)

    # ── Merge all three on subject ID ─────────────────────────────────────────
    merged = (survey_df
              .merge(eeg_df,   on="id", how="inner")
              .merge(watch_df, on="id", how="inner"))

    if len(merged) == 0:
        raise ValueError("Merge produced 0 rows — check subject ID formats match.")

    print(f"\n📦 Merged: {merged.shape[0]} subjects × {merged.shape[1]-2} features")
    print(f"   Subjects retained: {sorted(merged['id'].tolist())}")

    # ── Classify features: categorical (survey) vs continuous (EEG+Watch+Age) ─
    eeg_feat_cols   = [c for c in eeg_df.columns   if c != "id"]
    watch_feat_cols = [c for c in watch_df.columns if c != "id"]
    cont_survey_cols = ["Age"]   # only Age is truly continuous from survey

    # Categorical = survey features minus Age
    cat_cols  = [c for c in survey_feat_cols if c != "Age" and c in merged.columns]
    cont_cols = (
        [c for c in cont_survey_cols if c in merged.columns]
        + [c for c in eeg_feat_cols   if c in merged.columns]
        + [c for c in watch_feat_cols if c in merged.columns]
    )

    print(f"\n📋 Feature breakdown:")
    print(f"   Categorical (survey)   : {len(cat_cols)}")
    print(f"   Continuous  (Age)      : {len([c for c in cont_survey_cols if c in merged.columns])}")
    print(f"   Continuous  (EEG agg)  : {len([c for c in eeg_feat_cols if c in merged.columns])}")
    print(f"   Continuous  (Watch agg): {len([c for c in watch_feat_cols if c in merged.columns])}")
    print(f"   Total features         : {len(cat_cols) + len(cont_cols)}")

    # ── Build arrays ──────────────────────────────────────────────────────────
    # Categorical: integer arrays (already encoded 0-indexed)
    for c in cat_cols:
        merged[c] = merged[c].fillna(merged[c].mode()[0]).astype(int)
    X_cat = merged[cat_cols].values.astype(np.int64)

    # Continuous: fill NaN then scale
    for c in cont_cols:
        merged[c] = merged[c].fillna(merged[c].median())
    X_cont = merged[cont_cols].values.astype(np.float32)

    y = merged["pain_label"].values.astype(np.int64)

    # Category sizes for SAINT embedding tables (max value + 1 per column)
    cat_dims = [int(merged[c].max()) + 1 for c in cat_cols]

    # ── Train / test split ────────────────────────────────────────────────────
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_state)

    # ── Scale continuous features ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_cont_tr = scaler.fit_transform(X_cont[tr_idx])
    X_cont_te = scaler.transform(X_cont[te_idx])

    print(f"\n✂️  Train: {len(tr_idx)} subjects | Test: {len(te_idx)} subjects")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # With only 33 subjects drop_last=False to preserve all test samples
    train_loader = DataLoader(
        SAINTDataset(X_cat[tr_idx], X_cont_tr, y[tr_idx]),
        batch_size=min(batch_size, len(tr_idx)),
        shuffle=True, drop_last=False)
    test_loader = DataLoader(
        SAINTDataset(X_cat[te_idx], X_cont_te, y[te_idx]),
        batch_size=min(batch_size, len(te_idx)),
        shuffle=False)

    return train_loader, test_loader, {
        "cat_cols":    cat_cols,
        "cont_cols":   cont_cols,
        "cat_dims":    cat_dims,
        "n_cat":       len(cat_cols),
        "n_cont":      len(cont_cols),
        "n_classes":   3,
        "class_names": CLASS_NAMES,
        "scaler":      scaler,
        "subject_ids": merged["id"].tolist(),
        "eeg_feat_cols":   eeg_feat_cols,
        "watch_feat_cols": watch_feat_cols,
    }


if __name__ == "__main__":
    train_loader, test_loader, meta = prepare_data()
    x_cat, x_cont, labels = next(iter(train_loader))
    print(f"\n✅ Batch — x_cat:{x_cat.shape}  x_cont:{x_cont.shape}  labels:{labels.tolist()}")
    print(f"   cat_dims (first 5) : {meta['cat_dims'][:5]}")
    print(f"   EEG features       : {meta['eeg_feat_cols'][:3]} ...")
    print(f"   Watch features     : {meta['watch_feat_cols'][:3]} ...")