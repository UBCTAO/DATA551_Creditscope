# Milestone 4 Reflection — CreditScope

## What We Built

CreditScope is our credit risk model monitoring dashboard, built with Dash and Altair and deployed on Render. It has three tabs — Model Performance, Drift Analysis, and Data Quality — with a sidebar for filtering by date range, loan grade, feature, and alert threshold. All filters now update every chart, which was the biggest fix since M2.

The dashboard shows KPI cards that change color based on model health, an AUC trend line with threshold bands, a PSI heatmap, a calibration scatter plot, distribution comparisons, per-feature missing-rate charts, and a volume chart colored by default rate. We also added a collapsible guide and tooltips for AUC, PSI, and DQ so users unfamiliar with these terms can follow along.

## How We Addressed Feedback

The M2 TA feedback was really detailed and we are grateful for it. The biggest issue was that our filters were not connected to the charts, which cost us 5 points. We rewired all the callbacks, fixed the fullscreen layout, improved label formatting (no more raw variable names like `dti`), split missing-value charts into one per feature, and made the PSI heatmap respond to filters.

For M3, we collected feedback from 5 peers using a SUS-based survey. The SUS score was 76.5 out of 100 — a "B" grade, above the industry average of 68. People really liked the drift analysis views — the PSI heatmap and distribution comparison were called out as the most useful charts multiple times. The most common complaint was that the Overall Health card felt too vague. We fixed this by showing which metric triggered the status right on the badge, like "Warning — driven by PSI".

Reviewers also asked for bigger font sizes. We tuned them where we could, but with so many charts on screen there is a trade-off between readability and fitting everything without scrolling.

## What We Did Not Build

There were some features people asked for that we did not have time to build — alert history logs, cohort-level drill-down, PDF export, and benchmark model comparisons. These are good ideas but really production-level features that need backend storage and more time. For this project we focused on making the core monitoring workflow solid first.

## What We Learned

CreditScope is a more domain-specific project than a typical course dashboard. Terms like AUC, PSI, calibration gap, and default rate come from credit risk and model validation — not things most people know without a finance background. So we expected that usability scores on "needing to learn things" (SUS10) would run higher than average, and that is exactly what happened.

The key insight from user testing was that the main barrier is domain knowledge, not the interface. People found it easy to navigate and use the controls, but needed help understanding what the numbers mean. This confirmed that the Dashboard Guide and tooltips were the right investment. If we kept working on this, we would add short "how to read this" notes right next to each chart.
