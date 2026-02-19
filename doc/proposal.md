# Dashboard Proposal — Credit Risk Model Health Monitor

## Motivation and Purpose

**Our role:** A model risk analytics team within a mid-size financial institution.

**Target audience:** Credit risk managers and model validators responsible for ensuring that production credit scoring models remain reliable, fair, and compliant over time.

Financial institutions deploy credit scoring models to automate lending decisions worth millions of dollars each month. However, a model that performs well at launch can silently degrade as the applicant population shifts — economic downturns change borrower behavior, new demographics enter the market, or upstream data pipelines introduce quality issues. When model decay goes undetected, institutions face both financial losses from inaccurate risk assessments and regulatory scrutiny under frameworks such as the Federal Reserve's SR 11-7 guidance on model risk management.

Despite this, most monitoring today is either manual (quarterly spreadsheet reviews) or locked inside expensive enterprise platforms inaccessible to smaller teams. We propose building **CreditScope** — an interactive model monitoring dashboard that enables credit risk professionals to visually track model performance, detect population drift, and identify data quality anomalies across time. By providing at-a-glance health indicators with the ability to drill down into specific features and time periods, CreditScope bridges the gap between a static model report and a real-time production monitoring system.

## Description of the Data

We will use the **Lending Club Loan Dataset**, a publicly available dataset containing approximately **2.26 million loan records** issued between 2007 and 2018, with **151 variables** per loan. The dataset is released under a **CC0 (Public Domain)** license and is accessible via [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

For this project, we start from a curated 2012–2018 subset of approximately **100,000 loans** (selected to ensure consistent variable definitions after platform changes).  
For fast EDA iteration and visual prototyping, we then use a **stratified working sample of ~50,000 loans** that preserves key segment distributions (e.g., loan grade and issue period).


- **Target variable:** `loan_status` — whether the borrower fully repaid, defaulted, or was charged off. We will binarize this into a default indicator (1 = Charged Off / Default, 0 = Fully Paid).
- **Temporal key:** `issue_d` — the month and year each loan was issued, which serves as our natural time dimension for monitoring model performance across monthly cohorts.
- **Borrower demographics:** `annual_inc` (annual income), `emp_length` (employment length), `home_ownership`, `addr_state` (US state).
- **Credit profile:** `dti` (debt-to-income ratio), `revol_util` (revolving credit utilization), `fico_range_low` / `fico_range_high` (FICO scores at origination), `open_acc` (number of open credit lines), `delinq_2yrs` (delinquencies in last 2 years).
- **Loan characteristics:** `loan_amnt` (loan amount), `int_rate` (interest rate), `term` (36 or 60 months), `grade` / `sub_grade` (Lending Club's internal risk grade), `purpose` (debt consolidation, credit card, home improvement, etc.).

We will also derive several monitoring-specific variables during data preprocessing:

- `pred_default_prob` — predicted probability of default from a pre-trained logistic regression model (trained on 2012–2014 data, monitored on 2015–2018 data).
- `psi_score` — Population Stability Index per feature per monthly cohort, measuring distribution drift relative to the training period.
- `missing_rate` — percentage of missing values per feature per monthly cohort.

## Research Questions and Usage Scenarios

**Research Questions**

Our dashboard is designed to help answer the following questions:

- **RQ1:** How does the model’s AUC evolve over time, both overall and across key borrower segments (e.g., loan grade, loan purpose, home ownership)?

- **RQ2:** Which input features exhibit the highest population drift (measured by PSI) relative to the training period, and during which time periods do these shifts occur?

- **RQ3:** Are there any temporal patterns in data quality (e.g., missing rates, record counts) that could help explain observed changes in model performance?

- **RQ4:** For high-risk segments (e.g., loan grades D/E), how does drift in key features (such as `dti` or `revol_util`) relate to changes in default rates and AUC over time?


**Persona:** Sarah is a Model Risk Analyst at a regional bank. She oversees the credit scoring model used for personal loan approvals. Every month, she needs to verify that the model's predictions are still trustworthy before the quarterly regulatory review. She also needs to quickly investigate when something looks off — for example, if the model suddenly performs worse for a specific borrower segment.

**Usage scenario:**

When Sarah opens CreditScope, she first sees the **Overview tab** — a set of health indicator cards showing the model's current AUC, the maximum Population Stability Index (PSI) across all features, and an overall data quality score. Each card is color-coded: green if the metric is within acceptable bounds, yellow if it is approaching a warning threshold, and red if it has crossed into the alert zone. Below the cards, a time-series line chart shows how AUC has evolved month by month over the past three years, with horizontal threshold bands marking the "acceptable" (> 0.70) and "warning" (0.65–0.70) zones.

Sarah notices that AUC has been gradually declining over the last six months. She uses the **segment dropdown** to filter by loan grade and discovers that the decline is concentrated in Grade D and E loans—higher-risk borrowers. She switches to the **Drift Analysis tab** to explore which input features have shifted the most. A bar chart ranks all features by their PSI values for the most recent quarter, and she can compare the current distribution of any selected feature against the training-period baseline using an overlaid histogram. She finds that `dti` (debt-to-income ratio) and `revol_util` (revolving utilization) show the highest drift, with PSI values exceeding 0.25—an alert-level threshold in this prototype setting.


To rule out data pipeline issues, Sarah checks the **Data Quality tab**, which shows time-series charts of missing value rates and record counts per monthly cohort. She confirms that the data volumes and completeness look normal, so the drift is likely a genuine population shift rather than a data error.

Based on these findings, Sarah identifies that the model needs recalibration for the higher-risk segments and prepares a brief for the model governance committee, supported by screenshots from CreditScope showing the performance decline and feature drift patterns. She hypothesizes that recent changes in consumer debt behavior — possibly linked to macroeconomic shifts — have caused the model's training assumptions to become stale for these borrower segments, and recommends a targeted model refresh. 

In this prototype, threshold values are configurable defaults for monitoring design and will be calibrated with institutional policy and validation standards in later milestones.

(Our persona “Sarah” is loosely inspired by our teammate Wei Li, who has several years of experience in finance and holds the CFA charter.)
