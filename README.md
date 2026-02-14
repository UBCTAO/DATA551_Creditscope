# CreditScope — Credit Risk Model Health Monitor

> An interactive dashboard for monitoring credit scoring model performance, detecting population drift, and tracking data quality over time. Built with Python, Altair, and Dash.

**🚀 Interactive app implementation: Milestone 2** · **[📋 Proposal](proposal.md)** · **[📝 Reflections](doc/)**


---

## About

CreditScope is a model monitoring dashboard for credit risk managers and model validators. Using the public Lending Club loan dataset (2.26M loans, 2007–2018), it simulates a common workflow: a credit scoring model is trained on historical data and then monitored as new loan cohorts arrive.

For reproducible and fast Milestone 1 EDA/prototyping, we start from a curated 2012–2018 subset (~100k loans), then use a stratified working sample (~50k loans) that preserves key segment structure (e.g., loan grade and issue period).

The dashboard is designed to answer questions such as:

- How does model AUC evolve over time, overall and across key borrower segments (e.g., loan grades)?
- Which input features show the strongest population drift relative to the training baseline, and when do those shifts occur?
- Do changes in data quality (missingness, record counts) coincide with periods of weaker model performance?
- For higher-risk segments (e.g., grades D/E), how do feature shifts relate to changes in default rates and AUC?

Instead of a static PDF-style model report, CreditScope provides an interactive view of performance, drift, and data quality—closer to what a small risk team would use in practice.

---

## App Description

The app uses a **sidebar + main-content** layout with three main tabs.

### Left Sidebar — Controls

- **Date range slider**: Select the monitoring window (monthly cohorts, 2012–2018).
- **Segment filter dropdown**: Break down metrics by loan grade (A–G), loan purpose, or home ownership.
- **Feature selector dropdown**: Choose the input feature to inspect in the drift view.
- **Threshold toggle** (radio buttons): Switch between standard and conservative alert thresholds for PSI and AUC.

### Tab 1 — Overview (Model Health)

- KPI cards for **current AUC**, **max PSI**, and **data quality score** (optional aggregate health flag), with traffic-light status colors.
- **AUC time-series line chart** with configurable threshold bands.
- **PSI heatmap** summarizing drift intensity across features and time periods.

### Tab 2 — Drift Analysis

- **PSI bar chart** ranking monitored features by drift score for the selected period.
- **Distribution comparison chart** (overlaid histograms/densities) comparing the training baseline vs. selected cohort for a chosen feature.
- Linked interaction: clicking a PSI bar updates the distribution comparison plot.

### Tab 3 — Data Quality

- **Missing-rate time series** for key features over time.
- **Record-count bar chart** by monthly cohort to detect volume anomalies.

---

## App Sketch

The sketch below illustrates the planned layout and interactions (wireframe; not final UI):

![CreditScope Dashboard Sketch](doc/sketch.png)

---

## Installation & Running Locally

```bash
# Clone the repository
git clone https://github.com/ubco-mds-2025-labs/creditscope.git
cd creditscope

# Create environment
conda env create -f environment.yaml
conda activate creditscope

# Run the app
python src/app.py
```

## Data Notes

- **Source dataset:** Lending Club public dataset on Kaggle (2007–2018; ~2.26M records).
- **Project scope:** We restrict analysis to 2012–2018 to keep variable definitions consistent after platform/reporting changes.
- **Working data size:** Curated subset (~100k) → stratified working sample (~50k) for fast EDA and visual prototyping.
- **Why stratified sampling:** It preserves key segment distributions while reducing iteration time in Milestone 1.
- **Interpretation note:** In Milestone 1, we focus on relative temporal/segment patterns rather than exact population-level rates.
