# Reflection — Milestone 2

## What We Implemented

CreditScope is now a functional prototype with three main monitoring views built using Dash and Altair:

**Overview Tab:** Four KPI indicator cards (Model AUC, Max PSI, Data Quality Score, Overall Health) that dynamically update based on the selected date range and threshold settings. Each card uses traffic-light coloring (green/yellow/red) to communicate status at a glance. Below the cards, an AUC time-series chart shows model discrimination performance across monthly cohorts with 3-month rolling smoothing and configurable warning/alert threshold bands. A PSI heatmap provides a quarterly summary of feature drift intensity across all monitored features.

**Drift Analysis Tab:** A PSI bar chart ranks monitored features by their drift score for the latest quarter, with threshold reference lines. An overlaid density plot compares the training-period (2012–2014) baseline distribution against the monitoring-period (2015–2018) distribution for a user-selected feature.

**Data Quality Tab:** A missing-rate time-series tracks feature-level data completeness over monthly cohorts. A volume bar chart shows loan counts per cohort, color-coded by default rate, enabling detection of volume anomalies.

**Sidebar Controls:** Date range slider, grade segment filter dropdown, feature selector dropdown, and standard/conservative threshold radio buttons — all connected to the charts via Dash callbacks.

**Pre-computation Pipeline:** A separate `data_pipeline.py` trains a logistic regression baseline on 2012–2014 data, scores all cohorts, and pre-computes AUC, PSI, missing rates, and volume metrics as CSV files. This keeps the dashboard responsive since no model inference happens at runtime.

## What Is Not Yet Implemented

- **Inter-plot linking between the PSI bar chart and distribution chart** — clicking a PSI bar does not yet update the distribution overlay. Currently the feature selector dropdown controls this. We plan to add click-based linking in M4.
- **Segment-level PSI and missing rate breakdowns** — the grade filter currently affects only the AUC chart. Extending it to drift and DQ views is planned.
- **Deployment polish** — debug mode needs to be disabled, and some layout responsiveness on smaller screens can be improved.

## Strengths and Limitations

The dashboard effectively communicates model health through a clear three-pillar structure (performance, drift, data quality) that mirrors real-world model risk management workflows. The traffic-light KPI cards make it immediately obvious whether action is needed. The 3-month rolling AUC reduces noise from small monthly samples while preserving trend visibility.

A key limitation is that our baseline logistic regression achieves a training AUC of ~0.70, which is modest. This is expected for a simple model on this feature set and actually makes the monitoring exercise more interesting — the model is realistic rather than overfit. Another limitation is that our stratified sampling means absolute default rates should be interpreted cautiously.

## Future Improvements

For M4, we plan to: connect inter-plot interactivity (PSI bar → distribution), extend segment filtering to all tabs, add cohort granularity toggle (month/quarter/year), and polish the visual styling and layout alignment.
