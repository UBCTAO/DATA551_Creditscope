"""
app.py - creditscope model monitoring dashboard.
quarterly cohort aggregation. canadian bank regulatory visual language.
"""

import os
import numpy as np
import pandas as pd
import altair as alt
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# data + constants
# ---------------------------------------------------------------------
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed",
)
scored_df = pd.read_csv(
    os.path.join(DATA_DIR, "scored_loans.csv"),
    parse_dates=["cohort", "cohort_q"],
)

TRAIN_YEARS = {2012, 2013, 2014}
DRIFT_FEATURES = ["dti", "int_rate", "annual_inc", "loan_amnt"]
DQ_FEATURES = ["dti", "int_rate", "annual_inc", "loan_amnt"]

DISPLAY_NAMES = {
    "dti": "Debt-to-Income Ratio",
    "int_rate": "Interest Rate",
    "annual_inc": "Annual Income",
    "loan_amnt": "Loan Amount",
}
ABBR_EXPLAIN = {
    "AUC": "Area Under ROC Curve — measures how well the model separates defaulters from non-defaulters. Higher is better.",
    "PSI": "Population Stability Index — measures how much a feature's distribution has shifted from the training baseline. Lower is more stable.",
    "DQ": "Data Quality — percentage of non-missing values across key monitored features. Higher is better.",
}
THRESHOLDS = {
    "standard": {"auc_warning": 0.65, "auc_alert": 0.60, "psi_warning": 0.20, "psi_alert": 0.50},
    "conservative": {"auc_warning": 0.68, "auc_alert": 0.65, "psi_warning": 0.15, "psi_alert": 0.40},
}

# --- final palette ---
COLOR_PRIMARY = "#1f4e79"
COLOR_COMPARE = "#4f81bd"
COLOR_BG = "#f7f8fa"
COLOR_CARD = "#ffffff"
COLOR_SIDEBAR = "#f0f2f5"
COLOR_TEXT = "#1a2332"
COLOR_SECONDARY = "#6c757d"
COLOR_MUTED = "#6c757d"
COLOR_HEALTHY = "#2ca02c"
COLOR_WARNING = "#f39c12"
COLOR_ALERT = "#d62728"
COLOR_BORDER = "#dce0e5"
COLOR_GRID = "#eef0f3"
PSI_COLORS = {"Low": COLOR_HEALTHY, "Medium": COLOR_WARNING, "High": COLOR_ALERT}
GRADIENT_PALETTE = ["#8d133a", "#b9534e", "#d88456", "#edaa64", "#f6c872", "#f1de7f", "#e2e4bc"]
GRADIENT_PALETTE_REVERSED = list(reversed(GRADIENT_PALETTE))
VOLUME_RATE_DOMAIN = [0.00, 0.08, 0.15, 0.25, 0.35]
VOLUME_RATE_RANGE = ["#DCE6F1", "#7FA6D6", "#F2C14E", "#E67E22", "#C0392B"]
MISSING_FEATURE_COLORS = {
    "Debt-to-Income Ratio": COLOR_PRIMARY,
    "Interest Rate": COLOR_HEALTHY,
    "Annual Income": COLOR_ALERT,
    "Loan Amount": COLOR_WARNING,
}

ALL_GRADES = sorted(scored_df["grade"].dropna().unique().tolist())
QUARTERLY_COHORTS = sorted(pd.to_datetime(scored_df["cohort_q"].dropna().unique()))


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def quarter_label(ts):
    """Convert a timestamp-like value to a readable quarter label.

    Args:
        ts: A datetime-like value representing a cohort timestamp.

    Returns:
        str: Label in the format ``YYYY QN``.
    """
    ts = pd.Timestamp(ts)
    return f"{ts.year} Q{ts.quarter}"


def empty_chart_html(msg):
    """Build a lightweight HTML placeholder for empty chart states.

    Args:
        msg: Message shown to users when a chart has no data.

    Returns:
        str: HTML snippet rendered inside chart iframes.
    """
    return f"<div style='padding:12px;color:{COLOR_SECONDARY};'>{msg}</div>"


def altair_to_html(chart):
    """Serialize an Altair chart into a minimal embeddable HTML document.

    Args:
        chart: Altair chart object to render in an iframe.

    Returns:
        str: Self-contained HTML string with embedded Vega output.
    """
    inner = chart.to_html(fullhtml=False, embed_options={"actions": False})
    return (
        "<html><head><style>"
        "html, body {"
        "  margin: 0; padding: 0; overflow: hidden;"
        "  width: 100%; height: 100%;"
        "  background: transparent;"
        "}"
        ".vega-embed {"
        "  width: 100% !important;"
        "  height: 100% !important;"
        "  display: flex;"
        "  flex-direction: column;"
        "  justify-content: center;"
        "}"
        ".vega-embed summary { display: none !important; }"
        "</style></head><body>"
        f"{inner}"
        "</body></html>"
    )


def with_common_style(chart):
    """Apply shared visual style settings to an Altair chart.

    Args:
        chart: Altair chart object before final theme configuration.

    Returns:
        alt.Chart: Styled chart with shared axis, legend, and title config.
    """
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelFontSize=10, titleFontSize=11,
            titleColor=COLOR_TEXT, labelColor=COLOR_SECONDARY,
            gridColor=COLOR_GRID, domainColor=COLOR_BORDER,
            labelAngle=0, titleFontWeight="normal",
        )
        .configure_title(
            fontSize=14, anchor="start", color=COLOR_TEXT,
            fontWeight=600,
            subtitleColor=COLOR_SECONDARY, subtitleFontSize=11,
            subtitleFontWeight="normal",
        )
        .configure_legend(
            titleFontSize=10, labelFontSize=9,
            titleColor=COLOR_TEXT, labelColor=COLOR_SECONDARY,
        )
    )


def get_date_range(slider_value):
    """Map range-slider indices to concrete quarter timestamps.

    Args:
        slider_value: Two-item list from Dash range slider (start_idx, end_idx).

    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: Start and end quarter timestamps.
    """
    idx_start = max(0, min(int(round(slider_value[0])), len(QUARTERLY_COHORTS) - 1))
    idx_end = max(0, min(int(round(slider_value[1])), len(QUARTERLY_COHORTS) - 1))
    return pd.Timestamp(QUARTERLY_COHORTS[idx_start]), pd.Timestamp(QUARTERLY_COHORTS[idx_end])


def filtered_current_df(date_range, grade):
    """Filter monitoring-period rows by selected date range and grade.

    Args:
        date_range: Range-slider index pair defining quarter bounds.
        grade: Selected grade value or ``"all"``.

    Returns:
        pd.DataFrame: Filtered copy used by callback computations.
    """
    start, end = get_date_range(date_range)
    dff = scored_df[(scored_df["cohort_q"] >= start) & (scored_df["cohort_q"] <= end)]
    if grade != "all":
        dff = dff[dff["grade"] == grade]
    return dff.copy()


def filtered_baseline_df(grade):
    """Return training-baseline rows, optionally restricted by grade.

    Args:
        grade: Selected grade value or ``"all"``.

    Returns:
        pd.DataFrame: Baseline subset used as PSI reference distribution.
    """
    base = scored_df[scored_df["year"].isin(TRAIN_YEARS)]
    if grade != "all":
        bg = base[base["grade"] == grade]
        if len(bg) > 0:
            return bg.copy()
    return base.copy()


def safe_auc(grp):
    """Compute AUC safely for a cohort group with minimum data checks.

    Args:
        grp: Cohort-level dataframe slice with labels and predictions.

    Returns:
        float: AUC value, or ``np.nan`` when sample conditions are not met.
    """
    if len(grp) < 30 or grp["default_flag"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(grp["default_flag"], grp["pred_default_prob"]))


def compute_quarterly_auc(current_df):
    """Aggregate model discrimination metrics at quarter level.

    Args:
        current_df: Filtered monitoring dataframe.

    Returns:
        pd.DataFrame: Quarterly AUC, sample size, and default rate.
    """
    rows = []
    for q, grp in current_df.groupby("cohort_q"):
        auc = safe_auc(grp)
        if pd.notna(auc):
            rows.append({
                "quarter": pd.Timestamp(q), "quarter_label": quarter_label(q),
                "auc": round(auc, 4), "n_loans": len(grp),
                "default_rate": round(grp["default_flag"].mean(), 4),
            })
    return pd.DataFrame(rows).sort_values("quarter")


def compute_psi(expected, actual, n_bins=10):
    """Compute Population Stability Index between baseline and current arrays.

    Args:
        expected: Baseline numeric values (typically training period).
        actual: Current-period numeric values to compare against baseline.
        n_bins: Number of quantile bins derived from baseline values.

    Returns:
        float: PSI score, or ``np.nan`` when data is insufficient.
    """
    exp = expected[~np.isnan(expected)]
    act = actual[~np.isnan(actual)]
    if len(exp) < 20 or len(act) < 20:
        return np.nan
    edges = np.quantile(exp, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.nan
    eps = 1e-4
    exp_pct = np.histogram(exp, bins=edges)[0] / len(exp) + eps
    act_pct = np.histogram(act, bins=edges)[0] / len(act) + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_level(psi_val, threshold):
    """Map PSI value to Low/Medium/High drift risk level.

    Args:
        psi_val: Numeric PSI score.
        threshold: Threshold dictionary containing warning/alert PSI cutoffs.

    Returns:
        str: One of ``"Low"``, ``"Medium"``, or ``"High"``.
    """
    if psi_val < threshold["psi_warning"]:
        return "Low"
    if psi_val < threshold["psi_alert"]:
        return "Medium"
    return "High"


def compute_quarterly_psi(current_df, baseline_df, threshold):
    """Compute quarterly PSI for all monitored drift features.

    Args:
        current_df: Filtered current-period dataframe.
        baseline_df: Baseline dataframe used as expected distribution.
        threshold: Threshold dictionary for PSI severity buckets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
            Full quarterly PSI table, latest-quarter ranking table, and latest quarter.
    """
    rows = []
    for feat in DRIFT_FEATURES:
        expected = baseline_df[feat].dropna().values
        for q, grp in current_df.groupby("cohort_q"):
            actual = grp[feat].dropna().values
            psi = compute_psi(expected, actual)
            if pd.notna(psi):
                rows.append({
                    "quarter": pd.Timestamp(q), "quarter_label": quarter_label(q),
                    "feature": feat, "feature_display": DISPLAY_NAMES.get(feat, feat),
                    "psi": round(psi, 6),
                })
    out = pd.DataFrame(rows).sort_values(["quarter", "feature"])
    if len(out) == 0:
        return out, pd.DataFrame(), None
    out["psi_level"] = out["psi"].apply(lambda x: psi_level(x, threshold))
    latest_q = out["quarter"].max()
    latest = out[out["quarter"] == latest_q].sort_values("psi", ascending=False).copy()
    return out, latest, latest_q


def compute_quarterly_calibration(df):
    """Aggregate calibration metrics by quarter.

    Args:
        df: Filtered dataframe with labels and predicted probabilities.

    Returns:
        pd.DataFrame: Quarterly predicted rate, observed rate, and calibration gap.
    """
    rows = []
    for q, grp in df.groupby("cohort_q"):
        if len(grp) < 30:
            continue
        rows.append({
            "quarter": pd.Timestamp(q), "quarter_label": quarter_label(q),
            "avg_predicted": round(grp["pred_default_prob"].mean(), 4),
            "actual_default_rate": round(grp["default_flag"].mean(), 4),
            "n_loans": len(grp),
            "calibration_gap": round(grp["pred_default_prob"].mean() - grp["default_flag"].mean(), 4),
        })
    return pd.DataFrame(rows).sort_values("quarter")


def compute_quarterly_missing(current_df):
    """Compute missing-value rates per quarter and monitored feature.

    Args:
        current_df: Filtered dataframe for the selected monitoring scope.

    Returns:
        pd.DataFrame: Long-format table with quarterly missing rates.
    """
    rows = []
    for q, grp in current_df.groupby("cohort_q"):
        for feat in DQ_FEATURES:
            rows.append({
                "quarter": pd.Timestamp(q), "quarter_label": quarter_label(q),
                "feature_display": DISPLAY_NAMES.get(feat, feat),
                "missing_rate": round(grp[feat].isna().mean(), 6),
            })
    return pd.DataFrame(rows)


def compute_quarterly_volume(current_df):
    """Compute quarter-level loan volume and rate summary metrics.

    Args:
        current_df: Filtered dataframe for selected date range and grade.

    Returns:
        pd.DataFrame: Quarterly loan counts, default rate, and average loan amount.
    """
    out = current_df.groupby("cohort_q", as_index=False).agg(
        n_loans=("loan_amnt", "size"),
        default_rate=("default_flag", "mean"),
        avg_loan_amnt=("loan_amnt", "mean"),
    )
    out.rename(columns={"cohort_q": "quarter"}, inplace=True)
    out["quarter_label"] = out["quarter"].apply(quarter_label)
    return out.sort_values("quarter")


def kpi_severity(c):
    """Convert bootstrap color keys to comparable severity rank.

    Args:
        c: Bootstrap semantic color key (success, warning, danger).

    Returns:
        int: Ordered severity rank where higher means more severe.
    """
    return {"success": 0, "warning": 1, "danger": 2}[c]


def status_for_auc(v, t):
    """Assign KPI status label and color for AUC value.

    Args:
        v: AUC metric value.
        t: Threshold dictionary containing AUC warning/alert cutoffs.

    Returns:
        tuple[str, str, str]: Display text, bootstrap color key, border color.
    """
    if v >= t["auc_warning"]:
        return "Healthy", "success", COLOR_HEALTHY
    if v >= t["auc_alert"]:
        return "Warning", "warning", COLOR_WARNING
    return "Alert", "danger", COLOR_ALERT


def status_for_psi(v, t):
    """Assign KPI status label and color for PSI value.

    Args:
        v: PSI metric value.
        t: Threshold dictionary containing PSI warning/alert cutoffs.

    Returns:
        tuple[str, str, str]: Display text, bootstrap color key, border color.
    """
    if v < t["psi_warning"]:
        return "Healthy", "success", COLOR_HEALTHY
    if v < t["psi_alert"]:
        return "Warning", "warning", COLOR_WARNING
    return "Alert", "danger", COLOR_ALERT


def status_for_dq(v):
    """Assign KPI status label and color for data quality score.

    Args:
        v: Data quality metric (1 - missing rate).

    Returns:
        tuple[str, str, str]: Display text, bootstrap color key, border color.
    """
    if v >= 0.95:
        return "Healthy", "success", COLOR_HEALTHY
    if v >= 0.90:
        return "Warning", "warning", COLOR_WARNING
    return "Alert", "danger", COLOR_ALERT


def kpi_card_style(border_color):
    """Create consistent style dictionary for KPI cards.

    Args:
        border_color: Left-border color used to indicate KPI status.

    Returns:
        dict: Dash-compatible style configuration for KPI card wrapper.
    """
    return {
        "backgroundColor": COLOR_CARD,
        "border": f"1px solid {COLOR_BORDER}",
        "borderLeft": f"4px solid {border_color}",
        "borderRadius": "8px",
        "height": "100%",
    }


# ---------------------------------------------------------------------
# chart builders
# ---------------------------------------------------------------------
def build_auc_chart(auc_data, threshold, grade_label):
    """Build the AUC trend chart with threshold lines and annotations.

    Args:
        auc_data: Quarterly AUC summary dataframe.
        threshold: Threshold dictionary for warning/alert reference lines.
        grade_label: Human-readable grade segment label for chart title.

    Returns:
        alt.Chart: Styled Altair chart object.
    """
    t = threshold
    q_order = auc_data.sort_values("quarter")["quarter_label"].tolist()
    tick_vals = q_order[::2]
    if q_order[-1] not in tick_vals:
        tick_vals.append(q_order[-1])
    y_min = min(float(auc_data["auc"].min()) - 0.03, t["auc_alert"] - 0.02)
    y_max = float(auc_data["auc"].max()) + 0.03
    line = (
        alt.Chart(auc_data)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.5, color=COLOR_PRIMARY)
        .encode(
            x=alt.X(
                "quarter_label:O",
                title="Quarter",
                sort=q_order,
                axis=alt.Axis(labelAngle=-90, values=tick_vals, labelFontSize=7),
            ),
            y=alt.Y("auc:Q", title="Model AUC", scale=alt.Scale(domain=[y_min, y_max])),
            tooltip=[
                alt.Tooltip("quarter_label:O", title="Quarter"),
                alt.Tooltip("auc:Q", title="AUC", format=".4f"),
                alt.Tooltip("n_loans:Q", title="Loans"),
                alt.Tooltip("default_rate:Q", title="Default Rate", format=".1%"),
            ],
        )
    )
    warn_rule = alt.Chart(pd.DataFrame({"y": [t["auc_warning"]]})).mark_rule(color=COLOR_WARNING, strokeDash=[6, 4], strokeWidth=1.5).encode(y="y:Q")
    alert_rule = alt.Chart(pd.DataFrame({"y": [t["auc_alert"]]})).mark_rule(color=COLOR_ALERT, strokeDash=[6, 4], strokeWidth=1.5).encode(y="y:Q")
    warn_lbl = alt.Chart(pd.DataFrame({"y": [t["auc_warning"]], "text": [f"Warning ({t['auc_warning']:.2f})"]})).mark_text(align="right", dx=-5, dy=-8, fontSize=10, color=COLOR_WARNING).encode(y="y:Q", text="text:N")
    alert_lbl = alt.Chart(pd.DataFrame({"y": [t["auc_alert"]], "text": [f"Alert ({t['auc_alert']:.2f})"]})).mark_text(align="right", dx=-5, dy=-8, fontSize=10, color=COLOR_ALERT).encode(y="y:Q", text="text:N")
    last_pt = auc_data.tail(1)
    last_lbl = alt.Chart(last_pt).mark_text(align="left", dx=8, dy=-10, fontSize=12, fontWeight=600, color=COLOR_PRIMARY).encode(x=alt.X("quarter_label:O", sort=q_order), y="auc:Q", text=alt.Text("auc:Q", format=".3f"))
    chart = (line + warn_rule + alert_rule + warn_lbl + alert_lbl + last_lbl).properties(
        width="container",
        height=320,
        padding={"top": 40, "bottom": 50, "left": 60, "right": 40},
        title=alt.Title(
            f"AUC Over Time — {grade_label}",
            subtitle="Quarterly model discrimination performance",
            fontSize=13,
            subtitleFontSize=10,
            anchor="start",
        ),
    )
    return with_common_style(chart)


def build_psi_heatmap(psi_data, threshold, latest_q):
    """Build feature-drift heatmap across quarters.

    Args:
        psi_data: Long-format quarterly PSI dataframe.
        threshold: Threshold dictionary (kept for interface consistency).
        latest_q: Latest quarter used for star marker annotation.

    Returns:
        alt.Chart: Styled Altair heatmap.
    """
    data = psi_data.copy()
    data["feature_display"] = data["feature_display"].apply(lambda x: x.replace(" Ratio", ""))
    latest_label = quarter_label(latest_q)
    data["quarter_display"] = data["quarter_label"].apply(lambda x: f"{x} ★" if x == latest_label else x)
    q_order = [f"{ql} ★" if ql == latest_label else ql for ql in data.sort_values("quarter")["quarter_label"].unique()]
    tick_vals = [q for i, q in enumerate(q_order) if i % 2 == 0]
    if q_order[-1] not in tick_vals:
        tick_vals.append(q_order[-1])

    chart = (
        alt.Chart(data)
        .mark_rect(stroke="#ffffff", strokeWidth=1.5, cornerRadius=2)
        .encode(
            x=alt.X(
                "quarter_display:O",
                title="Quarter",
                sort=q_order,
                axis=alt.Axis(labelAngle=-90, values=tick_vals, labelFontSize=7),
            ),
            y=alt.Y("feature_display:N", title="Feature"),
            color=alt.Color(
                "psi:Q", title="PSI",
                scale=alt.Scale(
                    domain=[0, 0.1, 0.25, 0.5],
                    range=["#2d8a4e", "#f5d44b", "#e8763a", "#b5182b"],
                    clamp=True,
                ),
                legend=alt.Legend(
                    orient="right",
                    direction="vertical",
                    gradientLength=100,
                    gradientThickness=10,
                    title="PSI",
                    titleFontSize=9,
                    labelFontSize=8,
                    format=".2f",
                ),
            ),
            tooltip=[
                alt.Tooltip("feature_display:N", title="Feature"),
                alt.Tooltip("quarter_display:O", title="Quarter"),
                alt.Tooltip("psi:Q", title="PSI", format=".4f"),
            ],
        )
        .properties(
            width="container",
            height=160,
            title=alt.Title(
                "Feature Drift Heatmap (PSI)",
                subtitle=[
                    "Color = PSI severity; each quarter scored independently against baseline",
                    "Filters change which quarters are shown, not their values — ★ = latest quarter",
                ],
                fontSize=13,
                subtitleFontSize=10,
                anchor="start",
            ),
            padding={"left": 20, "right": 20, "top": 20, "bottom": 45},
        )
    )
    return with_common_style(chart)


def build_calibration_chart(cal_data):
    """Build calibration scatter chart with diagonal reference line.

    Args:
        cal_data: Quarterly calibration summary dataframe.

    Returns:
        alt.Chart: Styled Altair calibration chart.
    """
    data = cal_data.copy()
    data["period"] = data["quarter"].apply(lambda q: "Baseline (2012–2014)" if q.year in TRAIN_YEARS else "Monitoring (2015–2018)")
    x_min, x_max = data["avg_predicted"].min(), data["avg_predicted"].max()
    y_min, y_max = data["actual_default_rate"].min(), data["actual_default_rate"].max()
    axis_min = max(0, min(x_min, y_min) - 0.03)
    axis_max = max(x_max, y_max) + 0.03

    ref = alt.Chart(pd.DataFrame({"x": [axis_min, axis_max], "y": [axis_min, axis_max]})).mark_line(strokeDash=[6, 4], color=COLOR_MUTED, strokeWidth=1).encode(x="x:Q", y="y:Q")
    ref_lbl = alt.Chart(pd.DataFrame({"x": [(axis_min + axis_max) / 2], "y": [(axis_min + axis_max) / 2 + 0.012], "text": ["Perfect Calibration"]})).mark_text(fontSize=9, color=COLOR_MUTED, angle=40, fontStyle="italic").encode(x="x:Q", y="y:Q", text="text:N")
    pts = (
        alt.Chart(data).mark_circle(size=80, opacity=0.85)
        .encode(
            x=alt.X("avg_predicted:Q", title="Average Predicted Default Probability", scale=alt.Scale(domain=[axis_min, axis_max])),
            y=alt.Y("actual_default_rate:Q", title="Observed Default Rate", scale=alt.Scale(domain=[axis_min, axis_max])),
            color=alt.Color(
                "period:N",
                title="Period",
                scale=alt.Scale(
                    domain=["Baseline (2012–2014)", "Monitoring (2015–2018)"],
                    range=["#4878A8", "#E45756"],
                ),
                legend=alt.Legend(orient="right", direction="vertical"),
            ),
            tooltip=[alt.Tooltip("quarter_label:O", title="Quarter"), alt.Tooltip("avg_predicted:Q", title="Avg Predicted", format=".3f"), alt.Tooltip("actual_default_rate:Q", title="Actual Default Rate", format=".3f"), alt.Tooltip("calibration_gap:Q", title="Gap (Pred−Actual)", format="+.3f"), alt.Tooltip("n_loans:Q", title="Loans")],
        )
    )
    lbl_pts = pd.concat([data.head(1), data.tail(1)])
    lbls = alt.Chart(lbl_pts).mark_text(dx=12, dy=-8, fontSize=9, color=COLOR_SECONDARY).encode(x="avg_predicted:Q", y="actual_default_rate:Q", text="quarter_label:N")
    chart = (ref + ref_lbl + pts + lbls).properties(width="container", height="container",
        title=alt.Title("Model Calibration Over Time", subtitle="Points below diagonal = model overpredicts risk", anchor="start"))
    return with_common_style(chart)


def build_psi_bar(psi_latest, threshold, latest_q):
    """Build latest-quarter PSI ranking bar chart.

    Args:
        psi_latest: Latest-quarter PSI dataframe sorted by severity.
        threshold: Threshold dictionary defining PSI warning/alert bands.
        latest_q: Quarter shown in chart title.

    Returns:
        alt.Chart: Styled Altair bar chart with threshold guides.
    """
    t = threshold
    low_lbl = f"Low (< {t['psi_warning']:.2f})"
    med_lbl = f"Medium ({t['psi_warning']:.2f}–{t['psi_alert']:.2f})"
    high_lbl = f"High (≥ {t['psi_alert']:.2f})"
    data = psi_latest.copy()
    data["psi_bin"] = data["psi_level"].map({"Low": low_lbl, "Medium": med_lbl, "High": high_lbl})
    psi_max = float(data["psi"].max())
    y_upper = max(psi_max * 1.3, t["psi_warning"] * 1.2)
    show_alert = y_upper >= t["psi_alert"]

    bar = alt.Chart(data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("feature_display:N", title=None, sort="-y"),
        y=alt.Y("psi:Q", title="PSI", scale=alt.Scale(domain=[0, y_upper])),
        color=alt.Color("psi_bin:N", title="Risk Band", scale=alt.Scale(domain=[low_lbl, med_lbl, high_lbl], range=[COLOR_HEALTHY, COLOR_WARNING, COLOR_ALERT]), legend=None),
        tooltip=[alt.Tooltip("feature_display:N", title="Feature"), alt.Tooltip("psi:Q", title="PSI", format=".4f"), alt.Tooltip("psi_bin:N", title="Risk Band")],
    )
    bar_lbls = alt.Chart(data).mark_text(dy=-8, fontSize=10, fontWeight=600, color=COLOR_TEXT).encode(
        x=alt.X("feature_display:N", sort="-y"),
        y=alt.Y("psi:Q", scale=alt.Scale(domain=[0, y_upper])),
        text=alt.Text("psi:Q", format=".3f"),
    )
    wr = alt.Chart(pd.DataFrame({"y": [t["psi_warning"]]})).mark_rule(color=COLOR_WARNING, strokeDash=[6, 4], strokeWidth=1).encode(y="y:Q")
    ar = alt.Chart(pd.DataFrame({"y": [t["psi_alert"]]})).mark_rule(color=COLOR_ALERT, strokeDash=[6, 4], strokeWidth=1).encode(y="y:Q")
    wl = alt.Chart(pd.DataFrame({"y": [t["psi_warning"]], "text": ["Warning"]})).mark_text(align="right", dx=-5, dy=-8, fontSize=9, color=COLOR_WARNING).encode(y="y:Q", text="text:N")
    al = alt.Chart(pd.DataFrame({"y": [t["psi_alert"]], "text": ["Alert"]})).mark_text(align="right", dx=-5, dy=-8, fontSize=9, color=COLOR_ALERT).encode(y="y:Q", text="text:N")
    layers = [bar, bar_lbls, wr, wl]
    if show_alert:
        layers.extend([ar, al])
    chart = alt.layer(*layers).properties(
        width="container",
        height="container",
        title=alt.Title(
            f"Feature PSI Ranking — {quarter_label(latest_q)}",
            subtitle="Latest quarter vs training baseline",
            anchor="start",
        ),
    )
    return with_common_style(chart)


def build_distribution_chart(feature, current_df, baseline_df):
    """Build baseline-vs-current distribution comparison for one feature.

    Args:
        feature: Feature column name selected in the sidebar.
        current_df: Filtered current-period dataframe.
        baseline_df: Baseline dataframe used for comparison.

    Returns:
        alt.Chart | None: Distribution chart, or ``None`` when data is unavailable.
    """
    feat_display = DISPLAY_NAMES.get(feature, feature)
    cur_vals, base_vals = current_df[feature].dropna(), baseline_df[feature].dropna()
    if len(cur_vals) == 0 or len(base_vals) == 0:
        return None
    chart_data = pd.concat([
        pd.DataFrame({feature: base_vals, "period": "Baseline (2012–2014)"}),
        pd.DataFrame({feature: cur_vals, "period": "Current (selected)"}),
    ], ignore_index=True)
    low, high = chart_data[feature].quantile(0.01), chart_data[feature].quantile(0.99)
    chart_data = chart_data[(chart_data[feature] >= low) & (chart_data[feature] <= high)]
    if len(chart_data) > 8000:
        chart_data = chart_data.groupby("period", group_keys=False).apply(lambda g: g.sample(n=min(len(g), 4000), random_state=42)).reset_index(drop=True)

    bins = alt.Bin(maxbins=25)
    bh = alt.Chart(chart_data[chart_data["period"] == "Baseline (2012–2014)"]).mark_bar(opacity=0.55, color="#4878A8").encode(x=alt.X(f"{feature}:Q", bin=bins, title=feat_display), y=alt.Y("count():Q", title="Count"))
    ch = alt.Chart(chart_data[chart_data["period"] == "Current (selected)"]).mark_bar(fillOpacity=0, stroke="#E45756", strokeWidth=2).encode(x=alt.X(f"{feature}:Q", bin=bins, title=feat_display), y=alt.Y("count():Q", title="Count"))
    lgnd = alt.Chart(pd.DataFrame({"period": ["Baseline (2012–2014)", "Current (selected)"], "v": [0, 0]})).mark_point(opacity=0).encode(color=alt.Color("period:N", title="Period", scale=alt.Scale(domain=["Baseline (2012–2014)", "Current (selected)"], range=["#4878A8", "#E45756"]), legend=alt.Legend(orient="right", direction="vertical")))
    chart = (bh + ch + lgnd).properties(width="container", height="container",
        title=alt.Title(f"Distribution: {feat_display}", subtitle="Training baseline (filled) vs current selection (outlined)", anchor="start"))
    return with_common_style(chart)


def build_missing_single_chart(missing_data, feature_display):
    """Build a single-feature missing-rate trend chart.

    Args:
        missing_data: Long-format missing-rate dataframe across features/quarters.
        feature_display: Human-readable feature name to visualize.

    Returns:
        alt.Chart: Styled Altair line/area chart.
    """
    d = missing_data[missing_data["feature_display"] == feature_display].copy()
    q_order = d.sort_values("quarter")["quarter_label"].unique().tolist()
    tick_vals = q_order[::4] if len(q_order) > 12 else q_order
    line_color = MISSING_FEATURE_COLORS.get(feature_display, COLOR_PRIMARY)

    line = (
        alt.Chart(d)
        .mark_line(
            point=alt.OverlayMarkDef(size=30, filled=True),
            strokeWidth=2.2,
            color=line_color,
        )
        .encode(
            x=alt.X(
                "quarter_label:O",
                title=None,
                sort=q_order,
                axis=alt.Axis(labelAngle=-45, labelFontSize=8, values=tick_vals),
            ),
            y=alt.Y(
                "missing_rate:Q",
                title="Missing Rate",
                axis=alt.Axis(format=".2%", titleFontSize=10, labelFontSize=9),
            ),
            tooltip=[
                alt.Tooltip("feature_display:N", title="Feature"),
                alt.Tooltip("quarter_label:O", title="Quarter"),
                alt.Tooltip("missing_rate:Q", title="Missing Rate", format=".4%"),
            ],
        )
    )
    area = (
        alt.Chart(d)
        .mark_area(opacity=0.12, color=line_color)
        .encode(
            x=alt.X("quarter_label:O", sort=q_order),
            y=alt.Y("missing_rate:Q"),
        )
    )
    chart = (area + line).properties(
        width="container",
        height="container",
        title=alt.Title(
            f"Missing Value Rate — {feature_display}",
            subtitle="Quarterly trend for this feature",
            anchor="start",
        ),
    )
    return with_common_style(chart)


def build_volume_chart(vol_data):
    """Build quarterly loan-volume bar chart colored by default rate.

    Args:
        vol_data: Quarterly volume summary dataframe.

    Returns:
        alt.Chart: Styled Altair volume chart.
    """
    dr_min = vol_data["default_rate"].min()
    dr_max = vol_data["default_rate"].max()
    q_order = vol_data.sort_values("quarter")["quarter_label"].tolist()
    tick_vals = q_order[::2]
    if q_order[-1] not in tick_vals:
        tick_vals.append(q_order[-1])

    rate_domain = [0.00, 0.10, 0.20, 0.30, 0.35]
    rate_range = ["#2E8B57", "#A7C957", "#F2C14E", "#E67E22", "#C0392B"]

    chart = (
        alt.Chart(vol_data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "quarter_label:O",
                title="Quarter",
                sort=q_order,
                axis=alt.Axis(
                    labelAngle=-45,
                    values=tick_vals,
                    labelFontSize=10,
                    tickSize=4,
                ),
            ),
            y=alt.Y(
                "n_loans:Q",
                title="Loan Count",
                axis=alt.Axis(format=",.0f"),
            ),
            color=alt.Color(
                "default_rate:Q",
                title="Default Rate",
                scale=alt.Scale(
                    domain=rate_domain,
                    range=rate_range,
                    clamp=True,
                ),
                legend=alt.Legend(
                    format=".0%",
                    orient="bottom",
                    direction="horizontal",
                    gradientLength=150,
                    gradientThickness=8,
                    title=f"Default Rate\n({dr_min:.1%} – {dr_max:.1%})",
                    titleFontSize=9,
                    labelFontSize=8,
                ),
            ),
            tooltip=[
                alt.Tooltip("quarter_label:O", title="Quarter"),
                alt.Tooltip("n_loans:Q", title="Loans", format=",.0f"),
                alt.Tooltip("default_rate:Q", title="Default Rate", format=".2%"),
                alt.Tooltip("avg_loan_amnt:Q", title="Avg Loan Amount", format="$,.0f"),
            ],
        )
        .properties(
            width="container",
            height=500,
            title=alt.Title(
                "Loan Volume by Quarter",
                subtitle="Bar height shows loan count; color shows observed default rate",
                anchor="start",
            ),
            padding={"left": 50, "right": 18, "top": 20, "bottom": 65},
        )
    )
    return with_common_style(chart)


# ---------------------------------------------------------------------
# app + layout
# ---------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="CreditScope", suppress_callback_exceptions=True)
server = app.server


def make_kpi_card(title, value_id, badge_id, card_id, explanation):
    """Create a reusable KPI card component for the overview row.

    Args:
        title: KPI title text displayed at top of the card.
        value_id: Dash component id for the large numeric value.
        badge_id: Dash component id for the status badge.
        card_id: Dash component id for the card wrapper.
        explanation: Small helper text shown under the badge.

    Returns:
        dbc.Card: Bootstrap card component used in the KPI panel.
    """
    return dbc.Card(dbc.CardBody([
        html.P(title, className="mb-1 small fw-bold", style={"color": COLOR_TEXT}),
        html.H3(id=value_id, className="mb-1 fw-bold"),
        dbc.Badge(id=badge_id, className="mb-2"),
        html.Div(explanation, className="text-muted", style={"fontSize": "0.72rem", "lineHeight": "1.3"}),
    ]), id=card_id, className="text-center h-100", style=kpi_card_style(COLOR_BORDER))


guide_section = html.Div([
    dbc.Button("Dashboard Guide ▸", id="guide-toggle", color="light", size="sm", className="mb-0",
               style={"fontSize": "0.8rem", "color": COLOR_SECONDARY}),
    dbc.Collapse(
        html.Div([
            html.P([html.Strong("What is CreditScope? "), "A credit scoring model was trained on Lending Club loans from 2012–2014. This dashboard tracks whether the model remains reliable as new borrowers arrive, using AUC, PSI, and data quality."], className="mb-1", style={"fontSize": "0.82rem", "color": COLOR_TEXT}),
            html.P([html.Strong("How to read: "), "Start with Model Performance for the health verdict. Then inspect Drift Analysis for shifted features and calibration behavior. Finally validate Data Quality and volume trends."], className="mb-1", style={"fontSize": "0.82rem", "color": COLOR_TEXT}),
            html.P([html.Span("■ ", style={"color": COLOR_HEALTHY}), "Healthy  ", html.Span("■ ", style={"color": COLOR_WARNING}), "Review Needed  ", html.Span("■ ", style={"color": COLOR_ALERT}), "Action Required"], className="mb-0", style={"fontSize": "0.82rem"}),
        ], style={"padding": "10px 16px", "backgroundColor": COLOR_GRID, "borderRadius": "8px", "margin": "6px 16px 8px 16px"}),
        id="guide-collapse", is_open=False,
    ),
], style={"padding": "4px 16px 0 16px"})

marks = {0: quarter_label(QUARTERLY_COHORTS[0]), len(QUARTERLY_COHORTS) - 1: quarter_label(QUARTERLY_COHORTS[-1])}

sidebar = dbc.Card(dbc.CardBody([
    html.H6("Controls", className="fw-bold mb-3", style={"color": COLOR_TEXT}),
    html.Label("Date Range", className="fw-bold small mb-1", style={"color": COLOR_TEXT}),
    dcc.RangeSlider(id="date-range", min=0, max=len(QUARTERLY_COHORTS) - 1, step=1,
                    value=[0, len(QUARTERLY_COHORTS) - 1], marks=marks,
                    tooltip={"placement": "bottom", "always_visible": False}, className="mb-1"),
    html.Div(id="date-range-label", className="text-center mb-3",
             style={"fontSize": "0.75rem", "color": COLOR_SECONDARY}),
    html.Label("Segment Filter (Grade)", className="fw-bold small mb-1", style={"color": COLOR_TEXT}),
    dcc.Dropdown(id="grade-filter", options=[{"label": "All Grades", "value": "all"}] + [{"label": f"Grade {g}", "value": g} for g in ALL_GRADES], value="all", clearable=False, className="mb-3"),
    html.Label("Feature Selector", className="fw-bold small mb-1", style={"color": COLOR_TEXT}),
    dcc.Dropdown(id="feature-select", options=[{"label": DISPLAY_NAMES[f], "value": f} for f in DRIFT_FEATURES], value="dti", clearable=False, className="mb-3"),
    html.Label("Alert Threshold", className="fw-bold small mb-1", style={"color": COLOR_TEXT}),
    dbc.RadioItems(id="threshold-toggle", options=[{"label": "Standard", "value": "standard"}, {"label": "Conservative", "value": "conservative"}], value="standard", inline=True, className="mb-1"),
    html.Div([
        html.Span("Standard", style={"fontWeight": "600"}), " = default cutoffs", html.Br(),
        html.Span("Conservative", style={"fontWeight": "600"}), " = stricter warning/alert thresholds",
    ], style={"fontSize": "0.7rem", "color": COLOR_SECONDARY, "lineHeight": "1.4", "textAlign": "left"}, className="mb-3"),
    html.Hr(className="my-2"),
    html.Div([
        html.Span("AUC ℹ", id="tip-auc", className="small me-2", style={"cursor": "pointer"}),
        html.Span("PSI ℹ", id="tip-psi", className="small me-2", style={"cursor": "pointer"}),
        html.Span("DQ ℹ", id="tip-dq", className="small", style={"cursor": "pointer"}),
        dbc.Tooltip(ABBR_EXPLAIN["AUC"], target="tip-auc"),
        dbc.Tooltip(ABBR_EXPLAIN["PSI"], target="tip-psi"),
        dbc.Tooltip(ABBR_EXPLAIN["DQ"], target="tip-dq"),
    ], style={"color": COLOR_SECONDARY}),
]), className="h-100", style={"width": "250px", "minWidth": "250px", "height": "100%", "backgroundColor": COLOR_SIDEBAR, "border": "none", "borderRight": f"1px solid {COLOR_BORDER}", "overflowY": "hidden"})

MODEL_HEIGHT = "calc(100vh - 300px)"
HALF_HEIGHT = "calc((100vh - 190px) / 2)"
FULL_HEIGHT = "calc(100vh - 210px)"
QUARTER_HEIGHT = "calc((100vh - 220px) / 2)"

IF_BASE = {"width": "100%", "height": "100%", "border": "none", "overflow": "hidden"}
IF_MISSING = {
    "width": "100%",
    "height": "100%",
    "border": "1px solid #dce0e5",
    "borderRadius": "8px",
    "overflow": "hidden",
    "backgroundColor": "#ffffff",
}

tab_model = html.Div([
    html.P("Is the model still discriminating well between good and bad borrowers?", className="mb-2", style={"color": COLOR_SECONDARY, "fontSize": "0.88rem"}),
    dbc.Row([
        dbc.Col(make_kpi_card("Overall Health", "kpi-health", "kpi-health-badge", "card-kpi-health", "Composite of model, drift, and data health"), md=3),
        dbc.Col(make_kpi_card("Model AUC", "kpi-auc", "kpi-auc-badge", "card-kpi-auc", "Discrimination power — higher is better"), md=3),
        dbc.Col(make_kpi_card("Max PSI (Drift)", "kpi-psi", "kpi-psi-badge", "card-kpi-psi", "Feature drift severity — lower is more stable"), md=3),
        dbc.Col(make_kpi_card("Data Quality", "kpi-dq", "kpi-dq-badge", "card-kpi-dq", "Key feature completeness — higher is better"), md=3),
    ], className="g-3 mb-2"),
    html.Iframe(
        id="chart-auc-time",
        style={
            "width": "100%",
            "height": MODEL_HEIGHT,
            "border": "none",
            "overflow": "hidden",
        },
    ),
], style={"height": "100%", "overflow": "hidden", "display": "flex", "flexDirection": "column"})

tab_drift = html.Div([
    html.P("Which input features have drifted? Has model output calibration degraded?", className="mb-2", style={"color": COLOR_SECONDARY, "fontSize": "0.88rem"}),
    html.Div([
        html.Div(
            html.Iframe(id="chart-psi-heatmap", style=IF_BASE),
            style={"width": "50%", "height": HALF_HEIGHT, "padding": "0 4px 4px 0"},
        ),
        html.Div(
            html.Iframe(id="chart-calibration", style=IF_BASE),
            style={"width": "50%", "height": HALF_HEIGHT, "padding": "0 0 4px 4px"},
        ),
    ], style={"display": "flex"}),
    html.Div([
        html.Div(
            html.Iframe(id="chart-psi-bar", style=IF_BASE),
            style={"width": "50%", "height": HALF_HEIGHT, "padding": "4px 4px 0 0"},
        ),
        html.Div(
            html.Iframe(id="chart-drift-dist", style=IF_BASE),
            style={"width": "50%", "height": HALF_HEIGHT, "padding": "4px 0 0 4px"},
        ),
    ], style={"display": "flex"}),
], style={"height": "100%", "overflow": "hidden", "display": "flex", "flexDirection": "column"})

tab_dq = html.Div([
    html.P("Are there data completeness or volume issues that could explain apparent drift?", className="mb-2", style={"color": COLOR_SECONDARY, "fontSize": "0.88rem"}),
    html.Div([
        html.Div([
            html.Div([
                html.Div(
                    html.Iframe(id="chart-missing-dti", style=IF_MISSING),
                    style={"width": "50%", "height": QUARTER_HEIGHT, "padding": "0 4px 4px 0"},
                ),
                html.Div(
                    html.Iframe(id="chart-missing-int-rate", style=IF_MISSING),
                    style={"width": "50%", "height": QUARTER_HEIGHT, "padding": "0 0 4px 4px"},
                ),
            ], style={"display": "flex"}),
            html.Div([
                html.Div(
                    html.Iframe(id="chart-missing-annual-inc", style=IF_MISSING),
                    style={"width": "50%", "height": QUARTER_HEIGHT, "padding": "4px 4px 0 0"},
                ),
                html.Div(
                    html.Iframe(id="chart-missing-loan-amnt", style=IF_MISSING),
                    style={"width": "50%", "height": QUARTER_HEIGHT, "padding": "4px 0 0 4px"},
                ),
            ], style={"display": "flex"}),
        ], style={"width": "60%", "height": FULL_HEIGHT}),
        html.Div(
            html.Iframe(id="chart-volume", style=IF_BASE),
            style={"width": "40%", "height": FULL_HEIGHT, "paddingLeft": "8px"},
        ),
    ], style={"display": "flex", "flex": "1"}),
], style={"height": "100%", "overflow": "hidden", "display": "flex", "flexDirection": "column"})

app.layout = html.Div([
    html.Div(dbc.Container(dbc.Row([
        dbc.Col([html.H4("CreditScope", className="mb-0 fw-bold", style={"color": COLOR_TEXT, "letterSpacing": "0.5px"}), html.Small("Credit Risk Model Health Monitor – Lending Club 2012–2018", style={"color": COLOR_SECONDARY})], md=8),
        dbc.Col([html.Small("Baseline: Logistic Regression (2012–2014)", className="d-block text-end", style={"color": COLOR_SECONDARY}), html.Small("Current: Selected date range", className="d-block text-end", style={"color": COLOR_SECONDARY})], md=4),
    ], className="align-items-center"), fluid=True),
        style={"height": "56px", "borderBottom": f"1px solid {COLOR_BORDER}", "display": "flex", "alignItems": "center", "backgroundColor": COLOR_CARD}),
    guide_section,
    html.Div([
        html.Div(sidebar, style={"height": "100%"}),
        html.Div(dbc.Tabs([
            dbc.Tab(tab_model, label="Model Performance", tab_id="tab-model"),
            dbc.Tab(tab_drift, label="Drift Analysis", tab_id="tab-drift"),
            dbc.Tab(tab_dq, label="Data Quality", tab_id="tab-dq"),
        ], active_tab="tab-model"), style={"flex": "1", "height": "100%", "overflow": "hidden", "padding": "8px 12px"}),
    ], style={"flex": "1", "display": "flex", "overflow": "hidden"}),
], style={"height": "100vh", "display": "flex", "flexDirection": "column", "overflow": "hidden", "backgroundColor": COLOR_BG})


# ---------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------
@callback(Output("guide-collapse", "is_open"), Output("guide-toggle", "children"), Input("guide-toggle", "n_clicks"), State("guide-collapse", "is_open"))
def toggle_guide(n_clicks, is_open):
    """Toggle the dashboard guide collapse state and button label.

    Args:
        n_clicks: Number of times the guide toggle button was clicked.
        is_open: Current open/closed state of the guide section.

    Returns:
        tuple[bool, str]: New collapse state and updated button text.
    """
    if not n_clicks:
        return is_open, "Dashboard Guide ▸"
    new = not is_open
    return new, ("Dashboard Guide ▾" if new else "Dashboard Guide ▸")


@callback(Output("date-range-label", "children"), Input("date-range", "value"))
def update_date_label(date_range):
    """Render the selected quarter interval as a label under the slider.

    Args:
        date_range: Range-slider index pair (start_idx, end_idx).

    Returns:
        str: Human-readable range label in quarter format.
    """
    start, end = get_date_range(date_range)
    return f"{quarter_label(start)} — {quarter_label(end)}"


@callback(
    Output("kpi-health", "children"), Output("kpi-health-badge", "children"), Output("kpi-health-badge", "color"), Output("card-kpi-health", "style"),
    Output("kpi-auc", "children"), Output("kpi-auc-badge", "children"), Output("kpi-auc-badge", "color"), Output("card-kpi-auc", "style"),
    Output("kpi-psi", "children"), Output("kpi-psi-badge", "children"), Output("kpi-psi-badge", "color"), Output("card-kpi-psi", "style"),
    Output("kpi-dq", "children"), Output("kpi-dq-badge", "children"), Output("kpi-dq-badge", "color"), Output("card-kpi-dq", "style"),
    Input("date-range", "value"), Input("grade-filter", "value"), Input("threshold-toggle", "value"),
)
def update_kpis(date_range, grade, threshold_key):
    """Compute and format all KPI values, statuses, and card styles.

    Args:
        date_range: Range-slider index pair defining quarter bounds.
        grade: Selected grade segment or ``"all"``.
        threshold_key: Threshold profile key (``"standard"`` or ``"conservative"``).

    Returns:
        tuple: Values for KPI text, badge text/color, and card styles.
    """
    threshold = THRESHOLDS[threshold_key]
    current_df = filtered_current_df(date_range, grade)
    baseline_df = filtered_baseline_df(grade)
    auc_data = compute_quarterly_auc(current_df)
    _, psi_latest, _ = compute_quarterly_psi(current_df, baseline_df, threshold)
    missing_data = compute_quarterly_missing(current_df)

    auc_val = auc_data.iloc[-1]["auc"] if len(auc_data) else np.nan
    psi_val = psi_latest.iloc[0]["psi"] if len(psi_latest) else np.nan
    dq_val = (1.0 - missing_data[missing_data["quarter"] == missing_data["quarter"].max()]["missing_rate"].mean()) if len(missing_data) else np.nan

    auc_txt, auc_ck, auc_bc = status_for_auc(auc_val, threshold) if pd.notna(auc_val) else ("N/A", "warning", COLOR_WARNING)
    psi_txt, psi_ck, psi_bc = status_for_psi(psi_val, threshold) if pd.notna(psi_val) else ("N/A", "warning", COLOR_WARNING)
    dq_txt, dq_ck, dq_bc = status_for_dq(dq_val) if pd.notna(dq_val) else ("N/A", "warning", COLOR_WARNING)
    worst = max([auc_ck, psi_ck, dq_ck], key=kpi_severity)
    h_txt = {"success": "Healthy", "warning": "Warning", "danger": "Alert"}[worst]
    h_ck = worst
    h_bc = {0: COLOR_HEALTHY, 1: COLOR_WARNING, 2: COLOR_ALERT}[kpi_severity(worst)]
    triggers = []
    if auc_ck == worst:
        triggers.append("AUC")
    if psi_ck == worst:
        triggers.append("PSI")
    if dq_ck == worst:
        triggers.append("DQ")
    trigger_str = " + ".join(triggers)
    h_trigger = f"{h_txt} — driven by {trigger_str}"

    return (
        h_txt, h_trigger, h_ck, kpi_card_style(h_bc),
        f"{auc_val:.3f}" if pd.notna(auc_val) else "N/A", auc_txt, auc_ck, kpi_card_style(auc_bc),
        f"{psi_val:.3f}" if pd.notna(psi_val) else "N/A", psi_txt, psi_ck, kpi_card_style(psi_bc),
        f"{dq_val:.1%}" if pd.notna(dq_val) else "N/A", dq_txt, dq_ck, kpi_card_style(dq_bc),
    )


@callback(Output("chart-auc-time", "srcDoc"), Input("date-range", "value"), Input("grade-filter", "value"), Input("threshold-toggle", "value"))
def update_auc_chart(date_range, grade, threshold_key):
    """Update AUC time-series iframe content for current filters.

    Args:
        date_range: Range-slider index pair defining quarter bounds.
        grade: Selected grade segment or ``"all"``.
        threshold_key: Threshold profile key used for reference lines.

    Returns:
        str: HTML payload for the AUC chart iframe.
    """
    current_df = filtered_current_df(date_range, grade)
    auc_data = compute_quarterly_auc(current_df)
    if len(auc_data) == 0:
        return empty_chart_html("No AUC data available for current filters.")
    return altair_to_html(build_auc_chart(auc_data, THRESHOLDS[threshold_key], "All Grades" if grade == "all" else f"Grade {grade}"))


@callback(Output("chart-psi-heatmap", "srcDoc"), Input("date-range", "value"), Input("grade-filter", "value"), Input("threshold-toggle", "value"))
def update_psi_heatmap(date_range, grade, threshold_key):
    """Update PSI heatmap iframe content for current filters.

    Args:
        date_range: Range-slider index pair defining quarter bounds.
        grade: Selected grade segment or ``"all"``.
        threshold_key: Threshold profile key for drift interpretation.

    Returns:
        str: HTML payload for the PSI heatmap iframe.
    """
    threshold = THRESHOLDS[threshold_key]
    current_df = filtered_current_df(date_range, grade)
    baseline_df = filtered_baseline_df(grade)
    psi_data, _, latest_q = compute_quarterly_psi(current_df, baseline_df, threshold)
    if len(psi_data) == 0:
        return empty_chart_html("No PSI data available for current filters.")
    return altair_to_html(build_psi_heatmap(psi_data, threshold, latest_q))


@callback(
    Output("chart-calibration", "srcDoc"),
    Input("date-range", "value"),
    Input("grade-filter", "value"),
)
def update_calibration(date_range, grade):
    """Update calibration chart iframe content for current filters.

    Args:
        date_range: Range-slider index pair defining quarter bounds.
        grade: Selected grade segment or ``"all"``.

    Returns:
        str: HTML payload for the calibration chart iframe.
    """
    cal_df = filtered_current_df(date_range, grade)
    cal_data = compute_quarterly_calibration(cal_df)
    if len(cal_data) == 0:
        return empty_chart_html("No calibration data available for current filters.")
    return altair_to_html(build_calibration_chart(cal_data))


@callback(Output("chart-psi-bar", "srcDoc"), Input("date-range", "value"), Input("grade-filter", "value"), Input("threshold-toggle", "value"))
def update_psi_bar(date_range, grade, threshold_key):
    """Update latest-quarter PSI ranking iframe content.

    Args:
        date_range: Range-slider index pair defining quarter bounds.
        grade: Selected grade segment or ``"all"``.
        threshold_key: Threshold profile key for risk-band labeling.

    Returns:
        str: HTML payload for the PSI ranking iframe.
    """
    threshold = THRESHOLDS[threshold_key]
    current_df = filtered_current_df(date_range, grade)
    baseline_df = filtered_baseline_df(grade)
    _, latest, latest_q = compute_quarterly_psi(current_df, baseline_df, threshold)
    if len(latest) == 0:
        return empty_chart_html("No PSI ranking data available for current filters.")
    return altair_to_html(build_psi_bar(latest, threshold, latest_q))


@callback(Output("chart-drift-dist", "srcDoc"), Input("feature-select", "value"), Input("date-range", "value"), Input("grade-filter", "value"))
def update_drift_dist(feature, date_range, grade):
    """Update feature distribution comparison iframe content.

    Args:
        feature: Selected feature key from the feature dropdown.
        date_range: Range-slider index pair defining quarter bounds.
        grade: Selected grade segment or ``"all"``.

    Returns:
        str: HTML payload for the distribution comparison iframe.
    """
    current_df = filtered_current_df(date_range, grade)
    baseline_df = filtered_baseline_df(grade)
    if feature not in current_df.columns:
        return empty_chart_html("Feature not available.")
    chart = build_distribution_chart(feature, current_df, baseline_df)
    return altair_to_html(chart) if chart else empty_chart_html("No data for this filter.")


@callback(
    Output("chart-missing-dti", "srcDoc"),
    Output("chart-missing-int-rate", "srcDoc"),
    Output("chart-missing-annual-inc", "srcDoc"),
    Output("chart-missing-loan-amnt", "srcDoc"),
    Input("date-range", "value"),
    Input("grade-filter", "value"),
)
def update_missing_panels(date_range, grade):
    """Update all four missing-rate panel iframes for current filters.

    Args:
        date_range: Range-slider index pair defining quarter bounds.
        grade: Selected grade segment or ``"all"``.

    Returns:
        tuple[str, str, str, str]: HTML payloads for four missing-rate charts.
    """
    missing_data = compute_quarterly_missing(filtered_current_df(date_range, grade))
    if len(missing_data) == 0:
        empty = empty_chart_html("No missing-rate data available.")
        return empty, empty, empty, empty

    displays = [DISPLAY_NAMES[f] for f in DQ_FEATURES]
    html_blocks = []
    for disp in displays:
        chart = build_missing_single_chart(missing_data, disp)
        html_blocks.append(altair_to_html(chart))
    return tuple(html_blocks)


@callback(Output("chart-volume", "srcDoc"), Input("date-range", "value"), Input("grade-filter", "value"))
def update_volume(date_range, grade):
    """Update loan-volume iframe content for current filters.

    Args:
        date_range: Range-slider index pair defining quarter bounds.
        grade: Selected grade segment or ``"all"``.

    Returns:
        str: HTML payload for the volume chart iframe.
    """
    vol_data = compute_quarterly_volume(filtered_current_df(date_range, grade))
    if len(vol_data) == 0:
        return empty_chart_html("No volume data available.")
    return altair_to_html(build_volume_chart(vol_data))


if __name__ == "__main__":
    app.run(debug=False, port=8050)
