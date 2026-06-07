# Medicaid & Healthcare Access Risk Monitor
### State-Level Coverage Prioritization for Policy Teams and Program Officers

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chpghrwawmvddoquvmniwm.streamlit.app/)

---

## The Problem This Solves

> *Which states are under the most pressure on healthcare access — and what should happen next?*

State Medicaid program officers, federal policy teams, and healthcare coverage
administrators need a fast, evidence-based way to identify which states face the
highest access risk across insurance gaps, cost burden, income capacity, and
rural service reach — and to generate immediate, defensible policy directions.

This tool provides that: a composite risk score for all 50 US states, structured
policy briefs, priority banding, and a validated model — in a format that supports
real decision-making.

---

## What This Tool Produces

| Output | Description |
|---|---|
| **National Panel Snapshot** | High/medium/low priority state counts, average score, model match rate |
| **Focus State Brief** | Detailed risk explanation for any selected state — why it's positioned at its priority level |
| **Suggested Direction** | Concrete, priority-band-specific policy actions for coverage expansion, rural access, and cost reduction |
| **Full Policy Brief** | Expandable narrative-format brief for each state — briefing-document ready |
| **Access Risk Chart** | All 50 states ranked by score, color-coded by priority band with national average reference line |
| **Feature Importance Chart** | Model coefficient chart showing which indicators most drive risk scores |
| **State-by-State Table** | Full panel with all indicators, filterable by priority band |
| **CSV Export** | Full results table for inclusion in official planning documents |

---

## Access Risk Score — Method

The composite score blends four publicly available indicators, each normalized 0–100:

```
access_risk_score =
    uninsured_rate_norm     × 0.40   (primary driver: coverage gap)
  + rural_share_norm        × 0.25   (service delivery reach)
  + income_norm             × 0.20   (household capacity to afford care)
  + healthcare_cost_norm    × 0.15   (cost-of-care pressure)
```

**Priority bands:**
- 🔴 **High** — score ≥ 60: immediate coverage expansion and rural access investment needed
- 🟡 **Medium** — score 35–59: targeted outreach and preventive investment
- 🟢 **Low** — score < 35: sustain programs, monitor for emerging risk signals

**Model validation:**  
A logistic regression classifier assigns priority bands to held-out validation states,
achieving a **77% match rate** — indicating reliable signal for initial triage.
Production deployment should be validated against CMS enrollment data.

---

## Input Data Fields

| Field | Type | Description |
|---|---|---|
| `state` | string | US state name |
| `median_income` | float | Median household income (USD) |
| `uninsured_rate` | float | Share of population without health insurance (%) |
| `healthcare_cost_index` | float | Relative cost of healthcare vs. national baseline |
| `rural_population_share` | float | Share of population in rural areas (0–1) |

**Data sources for production deployment:**
- **Uninsured rate:** Census Bureau American Community Survey (ACS)
- **Median income:** ACS Table B19013
- **Healthcare cost index:** CMS Geographic Variation Public Use Files
- **Rural share:** USDA Economic Research Service rural-urban classifications

---

## Repository Structure

```
├── app.py          # Main Streamlit UI
├── modeling.py     # Access risk scoring and logistic regression
├── data.py         # Built-in state data and live data loader
├── config.py       # Configuration settings
├── policy.py       # Policy brief generation
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/S-ABDUL-AI/MEDICAID-HEALTHCARE-ACCESS-RISK-MONITOR.git
cd [REPO-NAME]

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

**Requirements:** `streamlit` · `pandas` · `numpy` · `plotly` · `scikit-learn`

---

## Deployment

Deployed on Streamlit Community Cloud.  
Live demo: [Medicaid & Healthcare Access Risk Monitor](https://chpghrwawmvddoquvmniwm.streamlit.app/)

**Rename the URL slug** (recommended):  
In Streamlit Cloud settings, change to: `medicaid-healthcare-access-risk-monitor`

---

## Why This Matters for Government Digital Services

This tool was built to demonstrate what government program officers actually need:
not just a chart, but a **structured decision brief** — risk, implication, and
action — that maps directly to the way policy decisions are made and documented.

The "Focus State Brief" section mirrors the output format of a real ministerial
or legislative brief. The "Suggested Direction" panel is priority-band-specific
and actionable within a single program cycle.

This design philosophy — building tools that help government workers make better
decisions for people who need them, rather than tools that impress data scientists —
reflects the author's approach to all products in this portfolio.

---

## Scope Note

> All built-in data is **illustrative** for product design demonstration.  
> Production Medicaid prioritization requires official CMS administrative data,
> ACS coverage statistics, state enrollment records, and legal programme review.  
> Model rankings should be validated against program-level data before use in
> official resource allocation decisions.

---

## Author

**Sherriff Abdul-Hamid**
Data Scientist · Development Economist · Public-Sector AI Researcher
[poverty360.org](https://poverty360.org) | [LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) | [GitHub](https://github.com/S-ABDUL-AI)

---

## Related Projects

| Project | Description |
|---|---|
| [Public Budget Allocation Tool](https://smart-resource-allocation-dashboard-eudzw5r2f9pbu4qyw3psez.streamlit.app/) | Need-based government budget distribution with ministerial brief — SNAP and Medicaid-adjacent program planning |
| [GovFund Allocation Engine](https://impact-allocation-engine-ahxxrbgwmvyapwmifahk2b.streamlit.app/) | Cost-effectiveness decision tool for global health funders — models cost-per-life-saved across malaria, nutrition, and social protection |
| [Community Vulnerability Index](https://povertyearlywarningsystem-7rrmkktbi7bwha2nna8gk7.streamlit.app/) | Predictive vulnerability targeting for safety net program outreach |
| [Global Vaccination Coverage Explorer](https://worldvaccinationcoverage-etl-ftvwbikifyyx78xyy2j3zv.streamlit.app/) | WHO vaccination data across 190+ countries — automated ETL for public health teams |
