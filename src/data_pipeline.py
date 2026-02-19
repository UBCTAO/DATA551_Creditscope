"""
data_pipeline.py — pre-computation pipeline for creditscope dashboard.

reads the 50k stratified sample, trains a logistic regression on 2012-2014,
scores all cohorts, computes monitoring metrics (auc, psi, missing rates),
and saves csvs for the dash app.

usage:  python src/data_pipeline.py
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# paths 
RAW_PATH = os.path.join("data", "raw", "lending_club_sample_2012_2018.csv")
OUT_DIR = os.path.join("data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# config 
TRAIN_YEARS = [2012, 2013, 2014]
MODEL_FEATURES = ["loan_amnt", "int_rate", "annual_inc", "dti", "term_months"]
DRIFT_FEATURES = ["dti", "int_rate", "annual_inc", "loan_amnt"]
DQ_FEATURES = ["dti", "annual_inc", "int_rate", "loan_amnt"]
PSI_N_BINS = 10
BAD_STATUSES = [
    "Charged Off", "Late (31-120 days)", "Late (16-30 days)", "Default"
]


def load_and_clean(path):
    """load raw csv and return cleaned dataframe with derived columns."""
    df = pd.read_csv(path, parse_dates=["issue_d"])
    print(f"loaded {len(df)} rows")

    # parse term to integer months (csv has leading spaces like " 36 months")
    df["term_months"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)

    # cap extreme dti (>100 is data quality flag)
    df.loc[df["dti"] > 100, "dti"] = np.nan

    # binary default flag
    df["default_flag"] = df["loan_status"].isin(BAD_STATUSES).astype(int)

    # monthly cohort
    df["cohort"] = df["issue_d"].dt.to_period("M").dt.to_timestamp()

    # quarterly cohort
    df["cohort_q"] = df["issue_d"].dt.to_period("Q").dt.to_timestamp()

    return df


def train_model(df):
    """train logistic regression on training-period data."""
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    train = train.dropna(subset=MODEL_FEATURES + ["default_flag"])

    X_train = train[MODEL_FEATURES].values
    y_train = train["default_flag"].values

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    print(f"training auc (2012-2014): {train_auc:.4f}")

    return model, imputer, scaler


def score_all(df, model, imputer, scaler):
    """add predicted default probability to all rows."""
    X = df[MODEL_FEATURES].values
    X = imputer.transform(X)
    X = scaler.transform(X)
    df["pred_default_prob"] = model.predict_proba(X)[:, 1]
    return df


def compute_psi(expected, actual, n_bins=PSI_N_BINS):
    """compute population stability index between two arrays."""
    edges = np.quantile(expected[~np.isnan(expected)], np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)

    e_counts = np.histogram(expected[~np.isnan(expected)], bins=edges)[0]
    a_counts = np.histogram(actual[~np.isnan(actual)], bins=edges)[0]

    eps = 1e-4
    e_pct = e_counts / e_counts.sum() + eps
    a_pct = a_counts / a_counts.sum() + eps

    psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return round(psi, 6)


def compute_auc_by_cohort(df):
    """compute auc per monthly and quarterly cohort."""
    results = []
    for gran, col in [("month", "cohort"), ("quarter", "cohort_q")]:
        for name, group in df.groupby(col):
            if group["default_flag"].nunique() < 2 or len(group) < 30:
                continue
            auc = roc_auc_score(group["default_flag"], group["pred_default_prob"])
            results.append({
                "period": name, "granularity": gran,
                "auc": round(auc, 4), "n_loans": len(group),
                "default_rate": round(group["default_flag"].mean(), 4),
            })
    auc_df = pd.DataFrame(results)

    # 3-month rolling for monthly
    monthly = auc_df[auc_df["granularity"] == "month"].sort_values("period").copy()
    monthly["auc_rolling3"] = monthly["auc"].rolling(3, min_periods=1).mean().round(4)
    auc_df = auc_df.merge(monthly[["period", "auc_rolling3"]], on="period", how="left")
    auc_df["auc_rolling3"] = auc_df["auc_rolling3"].fillna(auc_df["auc"])
    return auc_df


def compute_auc_by_cohort_grade(df):
    """compute auc per cohort x grade for segment drill-down."""
    results = []
    for (cohort, grade), group in df.groupby(["cohort", "grade"]):
        if group["default_flag"].nunique() < 2 or len(group) < 20:
            continue
        auc = roc_auc_score(group["default_flag"], group["pred_default_prob"])
        results.append({
            "period": cohort, "grade": grade,
            "auc": round(auc, 4), "n_loans": len(group),
            "default_rate": round(group["default_flag"].mean(), 4),
        })
    return pd.DataFrame(results)


def compute_psi_by_cohort(df):
    """compute psi per feature per monthly cohort vs training baseline."""
    train = df[df["year"].isin(TRAIN_YEARS)]
    results = []
    for feature in DRIFT_FEATURES:
        baseline = train[feature].dropna().values
        for cohort, group in df.groupby("cohort"):
            actual = group[feature].dropna().values
            if len(actual) < 10:
                continue
            psi_val = compute_psi(baseline, actual)
            results.append({
                "period": cohort, "feature": feature,
                "psi": psi_val, "n_obs": len(actual),
            })
    return pd.DataFrame(results)


def compute_missing_rates(df):
    """compute missing rate per feature per monthly cohort."""
    results = []
    for cohort, group in df.groupby("cohort"):
        for col in DQ_FEATURES:
            results.append({
                "period": cohort, "feature": col,
                "missing_rate": round(group[col].isna().mean(), 6),
                "n_records": len(group),
            })
    return pd.DataFrame(results)


def compute_volume_by_cohort(df):
    """record count and default rate per monthly cohort."""
    vol = df.groupby("cohort", as_index=False).agg(
        n_loans=("loan_amnt", "size"),
        default_rate=("default_flag", "mean"),
        avg_loan_amnt=("loan_amnt", "mean"),
    )
    vol.rename(columns={"cohort": "period"}, inplace=True)
    return vol


def compute_default_rate_by_grade(df):
    """default rate by grade overall."""
    return (
        df.groupby("grade", as_index=False)
        .agg(n_loans=("loan_amnt", "size"), default_rate=("default_flag", "mean"))
        .sort_values("grade")
    )


def compute_feature_distributions(df):
    """save training vs monitoring distributions for drift overlay chart."""
    df_out = df[DRIFT_FEATURES + ["year"]].copy()
    df_out["period_label"] = np.where(
        df_out["year"].isin(TRAIN_YEARS),
        "training (2012-14)", "monitoring (2015-18)"
    )
    sampled = []
    for label, group in df_out.groupby("period_label"):
        s = group.sample(n=min(len(group), 5000), random_state=42)
        sampled.append(s)
    return pd.concat(sampled, ignore_index=True)


def main():
    print("=" * 60)
    print("creditscope data pipeline")
    print("=" * 60)

    df = load_and_clean(RAW_PATH)
    print(f"cleaned: {len(df)} rows, default rate: {df['default_flag'].mean():.3f}")

    model, imputer, scaler = train_model(df)
    df_scored = score_all(df.copy(), model, imputer, scaler)

    print("\ncomputing metrics...")
    auc_df = compute_auc_by_cohort(df_scored)
    auc_grade_df = compute_auc_by_cohort_grade(df_scored)
    psi_df = compute_psi_by_cohort(df_scored)
    missing_df = compute_missing_rates(df)
    volume_df = compute_volume_by_cohort(df_scored)
    grade_df = compute_default_rate_by_grade(df_scored)
    dist_df = compute_feature_distributions(df)

    # save
    scored_cols = [
        "loan_amnt", "term_months", "int_rate", "grade", "sub_grade",
        "home_ownership", "annual_inc", "dti", "purpose",
        "default_flag", "cohort", "cohort_q", "year", "pred_default_prob"
    ]
    df_scored[scored_cols].to_csv(os.path.join(OUT_DIR, "scored_loans.csv"), index=False)
    auc_df.to_csv(os.path.join(OUT_DIR, "auc_by_cohort.csv"), index=False)
    auc_grade_df.to_csv(os.path.join(OUT_DIR, "auc_by_cohort_grade.csv"), index=False)
    psi_df.to_csv(os.path.join(OUT_DIR, "psi_by_cohort.csv"), index=False)
    missing_df.to_csv(os.path.join(OUT_DIR, "missing_rates.csv"), index=False)
    volume_df.to_csv(os.path.join(OUT_DIR, "volume_by_cohort.csv"), index=False)
    grade_df.to_csv(os.path.join(OUT_DIR, "default_rate_by_grade.csv"), index=False)
    dist_df.to_csv(os.path.join(OUT_DIR, "feature_distributions.csv"), index=False)

    print(f"\nall outputs saved to {OUT_DIR}/")
    for f in os.listdir(OUT_DIR):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f}: {size/1024:.0f} KB")
    print("pipeline complete.")


if __name__ == "__main__":
    main()
