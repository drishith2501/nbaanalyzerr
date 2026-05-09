# 🏀 NBA Playoff Game Predictor

A machine-learning pipeline that pulls **10 seasons of NBA playoff data** via the official `nba_api`, engineers advanced team-metric features, trains a **Logistic Regression** model, and predicts the winner of any playoff matchup.

---

## Tech Stack

| Library | Role |
|---|---|
| `nba_api` | Pull live & historical team stats and game logs |
| `pandas` | Data wrangling, feature engineering |
| `scikit-learn` | Logistic Regression, StandardScaler, CV evaluation |
| `seaborn` / `matplotlib` | EDA & diagnostic plots |
| `joblib` | Model persistence |

---

## Project Structure

```
NBA analyzer/
├── config.py               # All seasons, features, and file paths
├── data_collection.py      # Pull raw data from nba_api
├── data_preprocessing.py   # Engineer features & build training dataset
├── eda.py                  # Exploratory Data Analysis plots
├── model_training.py       # Train, evaluate, and save the model
├── predict.py              # Interactive CLI predictor
├── run_pipeline.py         # One-shot: run everything in order
├── requirements.txt
│
├── data/
│   ├── raw_team_stats.csv      # Advanced stats per team per season
│   ├── raw_game_log.csv        # Every playoff game row
│   └── processed_dataset.csv  # Final ML-ready dataset
│
├── models/
│   ├── logistic_regression_model.joblib
│   └── scaler.joblib
│
└── plots/
    ├── eda_home_win_rate.png
    ├── eda_diff_distributions.png
    ├── eda_correlation_heatmap.png
    ├── eda_boxplots.png
    ├── eda_rest_days.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.png
    └── prob_distribution.png
```

---

## Features Used by the Model

**Differential features** (home − away for each metric):
- `OFF_RATING_DIFF` – Offensive rating differential
- `DEF_RATING_DIFF` – Defensive rating differential
- `NET_RATING_DIFF` – Net rating differential
- `PACE_DIFF` – Pace differential
- `AST_PCT_DIFF` – Assist % differential
- `REB_PCT_DIFF` – Rebound % differential
- `TS_PCT_DIFF` – True shooting % differential
- `OPP_EFG_PCT_DIFF` – Opponent eFG% differential
- `OPP_TOV_PCT_DIFF` – Opponent TOV% differential

**Context features**:
- `HOME_REST_DAYS` / `AWAY_REST_DAYS` – Days of rest
- `REST_DIFF` – Rest advantage
- `HOME_SEED` / `AWAY_SEED` / `SEED_DIFF` – Playoff seeding
- `IS_HOME_HIGHER_SEED` – Binary seeding advantage flag

---

## Quick Start

### 1. Install dependencies
```bash
pip install --pre -r requirements.txt
```

### 2. Run the full pipeline (collect → preprocess → EDA → train)
```bash
python run_pipeline.py
```
> Data is cached — re-running skips the API calls unless you pass `--force`.

### 3. Predict a matchup
```bash
# Interactive
python predict.py

# Command-line flags
python predict.py --home BOS --away MIA --home-rest 3 --away-rest 2 --home-seed 2 --away-seed 3
```

### Sample output
```
═══════════════════════════════════════════════════════
  Playoff Matchup Prediction
═══════════════════════════════════════════════════════
  🏠 Home: Boston Celtics (#2 seed, 3d rest)
  ✈️  Away: Miami Heat (#3 seed, 2d rest)
─────────────────────────────────────────────────────
  ████████████████████████████░░░░░░░░░░░░
  BOS      70.4%   vs   MIA      29.6%
═══════════════════════════════════════════════════════

  ➡  Predicted winner: Boston Celtics  (70.4% confidence)
```

---

## Individual Scripts

| Script | Command |
|---|---|
| Data Collection | `python data_collection.py` |
| Preprocessing | `python data_preprocessing.py` |
| EDA | `python eda.py` |
| Model Training | `python model_training.py` |
| Predict a game | `python predict.py` |

---

## Model Details

- **Algorithm**: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Preprocessing**: StandardScaler (zero mean, unit variance)
- **Validation**: 5-fold stratified cross-validation on training seasons
- **Test set**: Most recent season held out (time-based split)
- **Metrics reported**: Accuracy, ROC-AUC, confusion matrix, classification report

---

## Notes

- The nba_api enforces rate limits. The pipeline includes a 1.5s delay between requests.
- Seeds are approximated by playoff wins when not directly available from the API.
- To retrain after adding more seasons, update `SEASONS` in `config.py` and run `python run_pipeline.py --force`.
