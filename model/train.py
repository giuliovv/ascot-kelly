"""
Ascot Kelly — XGBoost win-probability model + Benter second-stage combination.

Usage:
  python train.py --data /path/to/results.csv --predict ascot_saturday.csv

Outputs:
  model.json          — trained XGBoost ranker
  benter_coefs.json   — alpha (fundamental), gamma (market) for second-stage logit
  predictions.csv     — combined probabilities for tomorrow's runners
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Feature engineering ───────────────────────────────────────────────────────

def fractional_to_decimal(odds_str):
    """Convert '9/4' → 3.25. Returns NaN if unparseable."""
    try:
        if isinstance(odds_str, (int, float)):
            return float(odds_str)
        parts = str(odds_str).split("/")
        if len(parts) == 2:
            return float(parts[0]) / float(parts[1]) + 1
        return float(odds_str)
    except Exception:
        return np.nan


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns (adapt names to the actual Kaggle CSV):
      date, race_id, horse, position, dec_odds (or sp),
      going, distance, draw, jockey, trainer, weight, age, ...

    Returns df with feature columns added, sorted by (date, race_id, horse).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "race_id", "horse"]).reset_index(drop=True)

    # Numeric position (1 = winner; non-finishers → NaN)
    df["pos_num"] = pd.to_numeric(df["position"], errors="coerce")
    df["won"] = (df["pos_num"] == 1).astype(int)

    # De-vig market prob (multiplicative — quick & cheap for feature use)
    if "dec_odds" not in df.columns and "sp" in df.columns:
        df["dec_odds"] = df["sp"].apply(fractional_to_decimal)
    df["raw_prob"] = 1.0 / df["dec_odds"].clip(lower=1.01)
    race_sums = df.groupby("race_id")["raw_prob"].transform("sum")
    df["market_prob"] = df["raw_prob"] / race_sums

    # Field size
    df["field_size"] = df.groupby("race_id")["horse"].transform("count")

    # ── Rolling form features (computed *before* each race using past runs only) ──
    df = df.sort_values(["horse", "date", "race_id"]).reset_index(drop=True)

    for window in [3, 5, 10]:
        df[f"win_rate_{window}"] = (
            df.groupby("horse")["won"]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df[f"avg_pos_{window}"] = (
            df.groupby("horse")["pos_num"]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df[f"avg_mktprob_{window}"] = (
            df.groupby("horse")["market_prob"]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # Days since last run
    df["prev_date"] = df.groupby("horse")["date"].shift(1)
    df["days_since_run"] = (df["date"] - df["prev_date"]).dt.days.clip(upper=365)

    # Jockey & trainer win rates (rolling 3-month, computed naively for speed)
    for agent in ["jockey", "trainer"]:
        if agent in df.columns:
            df[f"{agent}_win_rate"] = (
                df.groupby(agent)["won"]
                .apply(lambda s: s.shift(1).expanding().mean())
                .reset_index(level=0, drop=True)
            )

    # Draw (barrier/stall position)
    if "draw" in df.columns:
        df["draw"] = pd.to_numeric(df["draw"], errors="coerce")

    # Weight
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Age
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Going encode
    if "going" in df.columns:
        df["going_enc"] = LabelEncoder().fit_transform(df["going"].fillna("Unknown"))

    return df


FEATURE_COLS = [
    "market_prob", "field_size",
    "win_rate_3", "win_rate_5", "win_rate_10",
    "avg_pos_3", "avg_pos_5", "avg_pos_10",
    "avg_mktprob_3", "avg_mktprob_5", "avg_mktprob_10",
    "days_since_run",
    # optional — included if present:
    "draw", "weight", "age", "going_enc",
    "jockey_win_rate", "trainer_win_rate",
]


def build_feature_matrix(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].astype(float).values
    return X, available


# ── Model ─────────────────────────────────────────────────────────────────────

def train_xgb_ranker(X_tr, y_tr, groups_tr):
    """XGBoost LambdaMART ranker (rank:pairwise), grouped by race."""
    model = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_tr, y_tr, group=groups_tr, verbose=False)
    return model


def calibrate_scores(scores, y, groups):
    """Platt scaling: fit a logistic regression on raw XGB scores → probabilities."""
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(scores.reshape(-1, 1), y)
    return lr


def benter_combination(fund_probs, market_probs):
    """
    Second-stage conditional logit: c_i = softmax(α·log(f_i) + γ·log(π_i))
    Fit α, γ by maximising log-likelihood on held-out data.
    Returns (alpha, gamma, combined_probs) for a single race.
    For training, call across many races and fit a single LR.
    """
    eps = 1e-9
    X = np.column_stack([
        np.log(np.clip(fund_probs, eps, 1)),
        np.log(np.clip(market_probs, eps, 1)),
    ])
    return X  # caller stacks across races then fits LR


# ── Shin de-vig (same logic as the JS tool) ───────────────────────────────────

def shin_devig(dec_odds):
    r = 1.0 / np.array(dec_odds, dtype=float)
    B = r.sum()
    pi = r / B
    for _ in range(300):
        z = (B - 1) / (B - (pi**2).sum())
        disc = z**2 + 4*(1-z)*r/B
        pi_new = (np.sqrt(disc) - z) / (2*(1-z))
        pi_new /= pi_new.sum()
        if np.abs(pi_new - pi).sum() < 1e-10:
            break
        pi = pi_new
    return pi


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to historical results CSV")
    parser.add_argument("--predict", default=None, help="CSV of tomorrow's runners (horse, dec_odds)")
    parser.add_argument("--out_dir", default="/home/ubuntu/ascot-kelly/model")
    parser.add_argument("--target_date", default="2026-06-20", help="Race date to predict")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    df_raw = pd.read_csv(args.data, low_memory=False)
    print(f"  {len(df_raw):,} rows, columns: {list(df_raw.columns)}")

    print("Engineering features…")
    df = engineer_features(df_raw)
    df_train = df[df["date"] < args.target_date].dropna(subset=["pos_num", "won"])
    print(f"  Training rows: {len(df_train):,}")

    X_all, feat_cols = build_feature_matrix(df_train)
    y_all = df_train["won"].values
    race_ids = df_train["race_id"].values

    # Time-ordered split: last 10% of races as validation
    unique_races = df_train["race_id"].unique()
    split_idx = int(len(unique_races) * 0.9)
    train_races = set(unique_races[:split_idx])
    val_races = set(unique_races[split_idx:])

    mask_tr = df_train["race_id"].isin(train_races).values
    mask_val = df_train["race_id"].isin(val_races).values

    X_tr, y_tr = X_all[mask_tr], y_all[mask_tr]
    X_val, y_val = X_all[mask_val], y_all[mask_val]

    groups_tr = df_train[mask_tr].groupby("race_id").size().values
    groups_val = df_train[mask_val].groupby("race_id").size().values

    print("Training XGBoost ranker…")
    model = train_xgb_ranker(X_tr, y_tr, groups_tr)
    model.save_model(str(out_dir / "model.json"))
    print("  Saved model.json")

    # Calibrate on validation set
    print("Calibrating…")
    scores_val = model.predict(X_val)
    platt = calibrate_scores(scores_val, y_val, groups_val)

    # Compute per-race fundamental probabilities on validation set
    df_val = df_train[mask_val].copy()
    df_val["xgb_score"] = scores_val
    df_val["fund_prob_raw"] = platt.predict_proba(scores_val.reshape(-1,1))[:,1]
    # Race-normalise
    df_val["fund_prob"] = df_val.groupby("race_id")["fund_prob_raw"].transform(lambda x: x / x.sum())
    df_val["market_prob_shin"] = df_val.groupby("race_id")["dec_odds"].transform(
        lambda x: shin_devig(x.values)
    ) if "dec_odds" in df_val.columns else df_val["market_prob"]

    # Fit Benter second-stage
    print("Fitting Benter second-stage logit…")
    rows = []
    labels = []
    for _, grp in df_val.groupby("race_id"):
        if len(grp) < 2:
            continue
        X_b = benter_combination(grp["fund_prob"].values, grp["market_prob"].values)
        rows.append(X_b)
        labels.extend(grp["won"].values)
    if rows:
        X_benter = np.vstack(rows)
        lr_benter = LogisticRegression(C=10.0, fit_intercept=False, max_iter=500)
        lr_benter.fit(X_benter, labels)
        alpha_b, gamma_b = lr_benter.coef_[0]
        print(f"  Benter coefs — α (fundamental)={alpha_b:.3f}, γ (market)={gamma_b:.3f}")
        coefs = {"alpha": alpha_b, "gamma": gamma_b}
    else:
        print("  Not enough val data for Benter — defaulting α=0.5, γ=0.5")
        coefs = {"alpha": 0.5, "gamma": 0.5}

    with open(out_dir / "benter_coefs.json", "w") as f:
        json.dump(coefs, f, indent=2)
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)
    with open(out_dir / "platt.json", "w") as f:
        json.dump({"coef": platt.coef_.tolist(), "intercept": platt.intercept_.tolist()}, f)

    print(f"  Artefacts saved to {out_dir}/")

    # ── Predict tomorrow ──────────────────────────────────────────────────────
    if args.predict:
        print(f"\nPredicting for {args.predict}…")
        df_pred = pd.read_csv(args.predict)
        # df_pred must have columns: race, horse, dec_odds, plus any features
        # Features for known horses are looked up from training history
        horse_features = {}
        for horse, grp in df_train.groupby("horse"):
            last = grp.sort_values("date").iloc[-1]
            horse_features[horse] = {c: last[c] for c in feat_cols if c in last.index}

        eps = 1e-9
        results = []
        for race_name, race_grp in df_pred.groupby("race"):
            horses = race_grp["horse"].tolist()
            dec_odds = race_grp["dec_odds"].tolist()
            market_probs = shin_devig(dec_odds)
            field_size = len(horses)

            fund_scores = []
            for i, horse in enumerate(horses):
                feats = horse_features.get(horse, {})
                row = np.array([
                    feats.get(c, market_probs[i] if c == "market_prob" else np.nan)
                    for c in feat_cols
                ], dtype=float)
                fund_scores.append(row)

            X_p = np.array(fund_scores)
            # Fill NaN with column medians from training
            col_medians = np.nanmedian(X_all, axis=0)
            for j in range(X_p.shape[1]):
                mask = np.isnan(X_p[:, j])
                X_p[mask, j] = col_medians[j]

            raw_scores = model.predict(X_p)
            raw_probs = platt.predict_proba(raw_scores.reshape(-1,1))[:,1]
            fund_probs = raw_probs / raw_probs.sum()

            # Benter combination
            log_f = np.log(np.clip(fund_probs, eps, 1))
            log_pi = np.log(np.clip(market_probs, eps, 1))
            combined_raw = np.exp(coefs["alpha"] * log_f + coefs["gamma"] * log_pi)
            combined_probs = combined_raw / combined_raw.sum()

            for i, horse in enumerate(horses):
                results.append({
                    "race": race_name,
                    "horse": horse,
                    "dec_odds": dec_odds[i],
                    "market_prob": round(market_probs[i] * 100, 2),
                    "fund_prob": round(fund_probs[i] * 100, 2),
                    "combined_prob": round(combined_probs[i] * 100, 2),
                    "edge_pct": round((combined_probs[i] * dec_odds[i] - 1) * 100, 2),
                })

        df_out = pd.DataFrame(results)
        out_path = out_dir / "predictions.csv"
        df_out.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")
        print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
