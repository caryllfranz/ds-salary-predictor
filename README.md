# DS Salary Intelligence

Predicts Data Science salaries based on role, experience, location, and company size.


Streamlit Dashboard

## What it does
- Cleans and processes real DS salary survey data (3,755 records)
- Trains an XGBoost model to predict salary in USD
- Shows which factors drive salary the most (SHAP)
- Interactive Streamlit dashboard with salary predictor

## Key Findings
- US companies pay **4x more** than Indian companies ($152K vs $38K)
- Senior roles earn **93% more** than entry level ($154K vs $79K)
- Medium companies pay more than large corporations ($142K vs $117K)
- DS salaries grew **46%** from 2020 to 2023

## Model Performance
| Model | MAE | R² |
|-------|-----|----|
| Linear Regression | Failed | — |
| Random Forest | $38,190 | 0.29 |
| **XGBoost ✅** | **$36,514** | **0.39** |

## Tech Stack
`Python` `pandas` `XGBoost` `SHAP` `Streamlit` `scikit-learn`

## How to Run
```bash
pip install -r requirements.txt
cd app
streamlit run streamlit_app.py
```

## Dataset
[Data Science Salaries 2023 — Kaggle](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)