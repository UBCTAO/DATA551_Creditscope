"""
app.py — creditscope interactive model monitoring dashboard.

a dash app with altair visualizations for credit risk model monitoring.
three tabs: overview, drift analysis, data quality.
sidebar controls: date range, segment filter, feature selector, threshold toggle.

usage:  python src/app.py
"""

import os
import json
import pandas as pd
import numpy as np
import altair as alt
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# load pre-computed data
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed"
)

auc_df = pd.read_csv(os.path.join(DATA_DIR, "auc_by_cohort.csv"), parse_dates=["period"])
auc_grade_df = pd.read_csv(os.path.join(DATA_DIR, "auc_by_cohort_grade.csv"), parse_dates=["period"])
psi_df = pd.read_csv(os.path.join(DATA_DIR, "psi_by_cohort.csv"), parse_dates=["period"])
missing_df = pd.read_csv(os.path.join(DATA_DIR, "missing_rates.csv"), parse_dates=["period"])
volume_df = pd.read_csv(os.path.join(DATA_DIR, "volume_by_cohort.csv"), parse_dates=["period"])
grade_df = pd.read_csv(os.path.join(DATA_DIR, "default_rate_by_grade.csv"))
dist_df = pd.read_csv(os.path.join(DATA_DIR, "feature_distributions.csv"))

# constants 
ALL_GRADES = sorted(auc_grade_df["grade"].unique())
DRIFT_FEATURES = sorted(psi_df["feature"].unique())

# date range from monthly auc data
monthly_auc = auc_df[auc_df["granularity"] == "month"].sort_values("period")
MIN_DATE = monthly_auc["period"].min()
MAX_DATE = monthly_auc["period"].max()

# threshold presets
THRESHOLDS = {
    "standard": {"auc_warning": 0.65, "auc_alert": 0.60, "psi_warning": 0.1, "psi_alert": 0.25},
    "conservative": {"auc_warning": 0.68, "auc_alert": 0.65, "psi_warning": 0.08, "psi_alert": 0.2},
}


# helper: altair chart to html
def altair_to_html(chart):
    """convert altair chart to html string for dash iframe."""
    return chart.to_html(fullhtml=False, embed_options={"actions": False})


# app setup 
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="CreditScope",
    suppress_callback_exceptions=True,
)
server = app.server

# sidebar
sidebar = dbc.Card(
    [
        html.H5("Controls", className="card-title mb-3",
                style={"fontWeight": "bold"}),

        # date range slider
        html.Label("Date Range", className="fw-bold small"),
        dcc.RangeSlider(
            id="date-range",
            min=0,
            max=len(monthly_auc) - 1,
            value=[0, len(monthly_auc) - 1],
            marks={
                0: monthly_auc["period"].iloc[0].strftime("%Y-%m"),
                len(monthly_auc) - 1: monthly_auc["period"].iloc[-1].strftime("%Y-%m"),
            },
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Hr(),

        # segment filter
        html.Label("Segment Filter (Grade)", className="fw-bold small"),
        dcc.Dropdown(
            id="grade-filter",
            options=[{"label": "All Grades", "value": "all"}]
            + [{"label": f"Grade {g}", "value": g} for g in ALL_GRADES],
            value="all",
            clearable=False,
            className="mb-3",
        ),

        # feature selector (for drift tab)
        html.Label("Feature Selector", className="fw-bold small"),
        dcc.Dropdown(
            id="feature-select",
            options=[{"label": f, "value": f} for f in DRIFT_FEATURES],
            value="dti",
            clearable=False,
            className="mb-3",
        ),

        # threshold toggle
        html.Label("Alert Thresholds", className="fw-bold small"),
        dbc.RadioItems(
            id="threshold-toggle",
            options=[
                {"label": "Standard", "value": "standard"},
                {"label": "Conservative", "value": "conservative"},
            ],
            value="standard",
            inline=True,
            className="mb-3",
        ),

        html.Hr(),
        html.P(
            "CreditScope monitors model health using AUC, PSI drift, "
            "and data quality metrics across monthly loan cohorts.",
            className="text-muted small",
        ),
    ],
    body=True,
    className="bg-light",
    style={"height": "100vh", "overflowY": "auto"},
)

# kpi cards (overview tab)
def make_kpi_card(title, value_id, color_id):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="card-title mb-1 small fw-bold"),
            html.H3(id=value_id, className="mb-0"),
            html.Small(id=color_id, className="text-muted"),
        ]),
        id=f"card-{value_id}",
        className="text-center shadow-sm",
    )


# tab content layouts
tab_overview = dbc.Container([
    # kpi cards row
    dbc.Row([
        dbc.Col(make_kpi_card("Model AUC", "kpi-auc", "kpi-auc-status"), md=3),
        dbc.Col(make_kpi_card("Max PSI (Drift)", "kpi-psi", "kpi-psi-status"), md=3),
        dbc.Col(make_kpi_card("Data Quality", "kpi-dq", "kpi-dq-status"), md=3),
        dbc.Col(make_kpi_card("Overall Health", "kpi-health", "kpi-health-status"), md=3),
    ], className="mb-4 g-3"),

    # auc time series
    dbc.Row([
        dbc.Col([
            html.Iframe(
                id="chart-auc-time",
                style={"width": "100%", "height": "460px", "border": "none"},
            )
        ], md=12),
    ], className="mb-4"),

    # psi heatmap
    dbc.Row([
        dbc.Col([
            html.Iframe(
                id="chart-psi-heatmap",
                style={"width": "100%", "height": "390px", "border": "none"},
            )
        ], md=12),
    ]),
], fluid=True)

tab_drift = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Iframe(
                id="chart-psi-bar",
                style={"width": "100%", "height": "430px", "border": "none"},
            )
        ], md=6),
        dbc.Col([
            html.Iframe(
                id="chart-drift-dist",
                style={"width": "100%", "height": "430px", "border": "none"},
            )
        ], md=6),
    ]),
], fluid=True)

tab_dq = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Iframe(
                id="chart-missing-rates",
                style={"width": "100%", "height": "430px", "border": "none"},
            )
        ], md=6),
        dbc.Col([
            html.Iframe(
                id="chart-volume",
                style={"width": "100%", "height": "430px", "border": "none"},
            )
        ], md=6),
    ]),
], fluid=True)


# main layout
app.layout = dbc.Container([
    # header
    dbc.Row([
        dbc.Col([
            html.H3("CreditScope", className="mb-0 fw-bold",
                    style={"color": "#2c3e50"}),
            html.Small("Credit Risk Model Health Monitor — Lending Club 2012–2018",
                      className="text-muted"),
        ], md=8),
        dbc.Col([
            html.Div([
                html.Small("Baseline: Logistic Regression (2012–14)",
                          className="text-muted d-block text-end"),
                html.Small("Monitoring: 2015–2018 monthly cohorts",
                          className="text-muted d-block text-end"),
            ])
        ], md=4),
    ], className="py-3 border-bottom mb-3"),

    # body: sidebar + tabs
    dbc.Row([
        dbc.Col(sidebar, md=3, className="pe-0"),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(tab_overview, label="Overview", tab_id="tab-overview",
                       activeTabClassName="fw-bold"),
                dbc.Tab(tab_drift, label="Drift Analysis", tab_id="tab-drift",
                       activeTabClassName="fw-bold"),
                dbc.Tab(tab_dq, label="Data Quality", tab_id="tab-dq",
                       activeTabClassName="fw-bold"),
            ], id="tabs", active_tab="tab-overview"),
        ], md=9),
    ]),
], fluid=True, className="px-4")


#  helper: get date range from slider
def get_date_range(slider_value):
    """convert slider indices to start/end timestamps."""
    start = monthly_auc["period"].iloc[slider_value[0]]
    end = monthly_auc["period"].iloc[slider_value[1]]
    return start, end


# callbacks

# kpi cards
@callback(
    Output("kpi-auc", "children"),
    Output("kpi-auc-status", "children"),
    Output("card-kpi-auc", "color"),
    Output("kpi-psi", "children"),
    Output("kpi-psi-status", "children"),
    Output("card-kpi-psi", "color"),
    Output("kpi-dq", "children"),
    Output("kpi-dq-status", "children"),
    Output("card-kpi-dq", "color"),
    Output("kpi-health", "children"),
    Output("kpi-health-status", "children"),
    Output("card-kpi-health", "color"),
    Input("date-range", "value"),
    Input("threshold-toggle", "value"),
)
def update_kpis(date_range, threshold_key):
    start, end = get_date_range(date_range)
    t = THRESHOLDS[threshold_key]

    # latest auc (monthly, most recent in range)
    m = monthly_auc[(monthly_auc["period"] >= start) & (monthly_auc["period"] <= end)]
    if len(m) > 0:
        latest_auc = m.iloc[-1]["auc_rolling3"]
    else:
        latest_auc = 0.5

    # max psi in range
    p = psi_df[(psi_df["period"] >= start) & (psi_df["period"] <= end)]
    if len(p) > 0:
        max_psi = p.groupby("period")["psi"].max().iloc[-1]
    else:
        max_psi = 0

    # data quality score (1 - avg missing rate across features)
    mdf = missing_df[(missing_df["period"] >= start) & (missing_df["period"] <= end)]
    if len(mdf) > 0:
        avg_missing = mdf.groupby("period")["missing_rate"].mean().iloc[-1]
        dq_score = 1 - avg_missing
    else:
        dq_score = 1.0

    # determine statuses
    def auc_status(v):
        if v >= t["auc_warning"]:
            return "Healthy", "success"
        elif v >= t["auc_alert"]:
            return "Warning", "warning"
        return "Alert", "danger"

    def psi_status(v):
        if v < t["psi_warning"]:
            return "Stable", "success"
        elif v < t["psi_alert"]:
            return "Warning", "warning"
        return "Alert", "danger"

    def dq_status(v):
        if v >= 0.95:
            return "Healthy", "success"
        elif v >= 0.90:
            return "Warning", "warning"
        return "Alert", "danger"

    auc_s, auc_c = auc_status(latest_auc)
    psi_s, psi_c = psi_status(max_psi)
    dq_s, dq_c = dq_status(dq_score)

    # overall health
    colors = [auc_c, psi_c, dq_c]
    if "danger" in colors:
        health_s, health_c = "ALERT", "danger"
    elif "warning" in colors:
        health_s, health_c = "Review", "warning"
    else:
        health_s, health_c = "Healthy", "success"

    return (
        f"{latest_auc:.3f}", auc_s, auc_c,
        f"{max_psi:.3f}", psi_s, psi_c,
        f"{dq_score:.1%}", dq_s, dq_c,
        health_s, "Overall Status", health_c,
    )


# auc time series chart
@callback(
    Output("chart-auc-time", "srcDoc"),
    Input("date-range", "value"),
    Input("grade-filter", "value"),
    Input("threshold-toggle", "value"),
)
def update_auc_chart(date_range, grade, threshold_key):
    start, end = get_date_range(date_range)
    t = THRESHOLDS[threshold_key]

    if grade == "all":
        # use overall monthly auc
        data = monthly_auc[
            (monthly_auc["period"] >= start) & (monthly_auc["period"] <= end)
        ].copy()

        line = (
            alt.Chart(data)
            .mark_line(point=True, strokeWidth=2, clip=True)
            .encode(
                x=alt.X("period:T", title="cohort (month)"),
                y=alt.Y("auc_rolling3:Q", title="AUC (3-month rolling)"),
                tooltip=[
                    alt.Tooltip("period:T", title="month"),
                    alt.Tooltip("auc:Q", title="AUC (point)", format=".3f"),
                    alt.Tooltip("auc_rolling3:Q", title="AUC (rolling 3m)", format=".3f"),
                    alt.Tooltip("n_loans:Q", title="loans"),
                ],
            )
        )
    else:
        data = auc_grade_df[
            (auc_grade_df["period"] >= start) &
            (auc_grade_df["period"] <= end) &
            (auc_grade_df["grade"] == grade)
        ].copy()

        line = (
            alt.Chart(data)
            .mark_line(point=True, strokeWidth=2, clip=True)
            .encode(
                x=alt.X("period:T", title="cohort (month)"),
                y=alt.Y("auc:Q", title="AUC"),
                tooltip=[
                    alt.Tooltip("period:T", title="month"),
                    alt.Tooltip("auc:Q", title="AUC", format=".3f"),
                    alt.Tooltip("n_loans:Q", title="loans"),
                    alt.Tooltip("default_rate:Q", title="default rate", format=".2%"),
                ],
            )
        )

    # threshold bands
    warning_rule = (
        alt.Chart(pd.DataFrame({"y": [t["auc_warning"]]}))
        .mark_rule(color="orange", strokeDash=[5, 3], strokeWidth=1.5)
        .encode(y="y:Q")
    )
    alert_rule = (
        alt.Chart(pd.DataFrame({"y": [t["auc_alert"]]}))
        .mark_rule(color="red", strokeDash=[5, 3], strokeWidth=1.5)
        .encode(y="y:Q")
    )

    grade_label = f" — Grade {grade}" if grade != "all" else ""
    chart = (
        (line + warning_rule + alert_rule)
        .properties(
            width="container", height=380,
            padding={"left": 10, "right": 10, "top": 10, "bottom": 45},
            title=f"AUC Over Time{grade_label}"
        )
        .configure_title(fontSize=14, anchor="start")
    )

    return altair_to_html(chart)


# psi heatmap
@callback(
    Output("chart-psi-heatmap", "srcDoc"),
    Input("date-range", "value"),
    Input("threshold-toggle", "value"),
)
def update_psi_heatmap(date_range, threshold_key):
    start, end = get_date_range(date_range)
    t = THRESHOLDS[threshold_key]

    data = psi_df[
        (psi_df["period"] >= start) & (psi_df["period"] <= end)
    ].copy()

    # aggregate to quarterly for cleaner heatmap
    data["quarter"] = data["period"].dt.to_period("Q").dt.to_timestamp()
    heat_data = data.groupby(["quarter", "feature"], as_index=False)["psi"].mean()
    heat_data["psi"] = heat_data["psi"].round(4)

    chart = (
        alt.Chart(heat_data)
        .mark_rect()
        .encode(
            x=alt.X("quarter:T", title="quarter"),
            y=alt.Y("feature:N", title="feature"),
            color=alt.Color(
                "psi:Q", title="PSI",
                scale=alt.Scale(
                    scheme="redyellowgreen", reverse=True,
                    domain=[0, t["psi_alert"]]
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("quarter:T", title="quarter"),
                alt.Tooltip("psi:Q", title="PSI", format=".4f"),
            ],
        )
        .properties(
            width="container", height=280,
            padding={"left": 10, "right": 10, "top": 10, "bottom": 45},
            title="Feature Drift Heatmap (PSI) — quarterly average"
        )
        .configure_title(fontSize=14, anchor="start")
    )

    return altair_to_html(chart)


# psi bar chart (drift tab)
@callback(
    Output("chart-psi-bar", "srcDoc"),
    Input("date-range", "value"),
    Input("threshold-toggle", "value"),
)
def update_psi_bar(date_range, threshold_key):
    start, end = get_date_range(date_range)
    t = THRESHOLDS[threshold_key]

    data = psi_df[
        (psi_df["period"] >= start) & (psi_df["period"] <= end)
    ].copy()

    # latest quarter psi per feature
    data["quarter"] = data["period"].dt.to_period("Q").dt.to_timestamp()
    latest_q = data["quarter"].max()
    latest = data[data["quarter"] == latest_q].groupby("feature", as_index=False)["psi"].mean()
    latest["psi"] = latest["psi"].round(4)

    warning_rule = (
        alt.Chart(pd.DataFrame({"y": [t["psi_warning"]]}))
        .mark_rule(color="orange", strokeDash=[5, 3], strokeWidth=1.5)
        .encode(y="y:Q")
    )
    alert_rule = (
        alt.Chart(pd.DataFrame({"y": [t["psi_alert"]]}))
        .mark_rule(color="red", strokeDash=[5, 3], strokeWidth=1.5)
        .encode(y="y:Q")
    )

    bar = (
        alt.Chart(latest)
        .mark_bar()
        .encode(
            x=alt.X(
                "feature:N",
                title="feature",
                sort="-y",
                axis=alt.Axis(labelAngle=0, labelLimit=140),
            ),
            y=alt.Y("psi:Q", title="PSI"),
            color=alt.condition(
                alt.datum.psi > t["psi_warning"],
                alt.value("#e74c3c"),
                alt.value("#3498db"),
            ),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("psi:Q", title="PSI", format=".4f"),
            ],
        )
    )

    chart = (
        (bar + warning_rule + alert_rule)
        .properties(
            width="container", height=350,
            padding={"left": 10, "right": 10, "top": 10, "bottom": 45},
            title="Feature PSI — Latest Quarter vs Training Baseline"
        )
        .configure_title(fontSize=14, anchor="start")
    )

    return altair_to_html(chart)


# drift distribution chart
@callback(
    Output("chart-drift-dist", "srcDoc"),
    Input("feature-select", "value"),
)
def update_drift_dist(feature):
    data = dist_df[dist_df[feature].notna()].copy()

    # clip to 1st-99th percentile to handle outliers
    low = data[feature].quantile(0.01)
    high = data[feature].quantile(0.99)
    data = data[(data[feature] >= low) & (data[feature] <= high)].copy()
    if data.empty:
        return "<div style='padding:12px;color:#666;'>No data available for this feature.</div>"

    # altair default inline data transformer limits rows to 5000
    plot_data = data.sample(n=5000, random_state=42) if len(data) > 5000 else data

    try:
        chart = (
            alt.Chart(plot_data)
            .transform_density(
                feature,
                as_=[feature, "density"],
                groupby=["period_label"],
                extent=[float(low), float(high)],
            )
            .mark_area(opacity=0.45, interpolate="monotone")
            .encode(
                x=alt.X(f"{feature}:Q", title=feature),
                y=alt.Y("density:Q", title="density"),
                color=alt.Color(
                    "period_label:N", title="period",
                    scale=alt.Scale(
                        domain=["training (2012-14)", "monitoring (2015-18)"],
                        range=["#3498db", "#e74c3c"],
                    ),
                ),
                tooltip=[alt.Tooltip("period_label:N", title="period")],
            )
            .properties(
                width="container", height=350,
                padding={"left": 10, "right": 10, "top": 10, "bottom": 45},
                title=f"Distribution Drift: {feature}"
            )
            .configure_title(fontSize=14, anchor="start")
        )
        return altair_to_html(chart)
    except Exception:
        # fallback to histogram if density fails
        chart = (
            alt.Chart(plot_data)
            .mark_bar(opacity=0.5)
            .encode(
                x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=30), title=feature),
                y=alt.Y("count():Q", title="count", stack=None),
                color=alt.Color(
                    "period_label:N", title="period",
                    scale=alt.Scale(
                        domain=["training (2012-14)", "monitoring (2015-18)"],
                        range=["#3498db", "#e74c3c"],
                    ),
                ),
            )
            .properties(
                width="container", height=350,
                padding={"left": 10, "right": 10, "top": 10, "bottom": 45},
                title=f"Distribution Drift: {feature}"
            )
            .configure_title(fontSize=14, anchor="start")
        )
        try:
            return altair_to_html(chart)
        except Exception:
            return (
                "<div style='padding:12px;color:#666;'>"
                "Unable to render drift distribution for this feature."
                "</div>"
            )


# missing rates chart (dq tab)
@callback(
    Output("chart-missing-rates", "srcDoc"),
    Input("date-range", "value"),
)
def update_missing_rates(date_range):
    start, end = get_date_range(date_range)

    data = missing_df[
        (missing_df["period"] >= start) & (missing_df["period"] <= end)
    ].copy()

    chart = (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X("period:T", title="cohort (month)"),
            y=alt.Y("missing_rate:Q", title="missing rate",
                    axis=alt.Axis(format=".2%")),
            color=alt.Color("feature:N", title="feature"),
            tooltip=[
                alt.Tooltip("period:T", title="month"),
                alt.Tooltip("feature:N"),
                alt.Tooltip("missing_rate:Q", title="missing rate", format=".3%"),
                alt.Tooltip("n_records:Q", title="records"),
            ],
        )
        .properties(
            width="container", height=350,
            padding={"left": 10, "right": 10, "top": 10, "bottom": 45},
            title="Missing Value Rates Over Time"
        )
        .configure_title(fontSize=14, anchor="start")
    )

    return altair_to_html(chart)


# volume chart (dq tab)
@callback(
    Output("chart-volume", "srcDoc"),
    Input("date-range", "value"),
)
def update_volume(date_range):
    start, end = get_date_range(date_range)

    data = volume_df[
        (volume_df["period"] >= start) & (volume_df["period"] <= end)
    ].copy()

    bar = (
        alt.Chart(data)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("period:T", title="cohort (month)"),
            y=alt.Y("n_loans:Q", title="loan count"),
            color=alt.Color(
                "default_rate:Q", title="default rate",
                scale=alt.Scale(scheme="redyellowgreen", reverse=True),
            ),
            tooltip=[
                alt.Tooltip("period:T", title="month"),
                alt.Tooltip("n_loans:Q", title="loans"),
                alt.Tooltip("default_rate:Q", title="default rate", format=".2%"),
                alt.Tooltip("avg_loan_amnt:Q", title="avg loan amt", format="$,.0f"),
            ],
        )
        .properties(
            width="container", height=350,
            padding={"left": 10, "right": 10, "top": 10, "bottom": 45},
            title="Loan Volume by Monthly Cohort"
        )
        .configure_title(fontSize=14, anchor="start")
    )

    return altair_to_html(bar)


# run
if __name__ == "__main__":
    app.run(debug=True, port=8050)
