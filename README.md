# Short-term traffic flow prediction on E4 Essingeleden (AH2179, KTH)

Predict the traffic flow (veh/h) at MCS portal **55.620** on the southbound E4 Essingeleden (Stockholm)
**15 minutes ahead, every 5 minutes of the day**, from one-minute radar data of eight portals
(58.140 → 55.620). The pipeline is built for real-time use: every model input uses only data available at
prediction time (tested in `A1`, section 4).

## Results (final hold-out, 101 days of 2022, used once)

| Model | RMSE all day (veh/h) | RMSE 07:30–08:30 (veh/h) |
|---|---|---|
| Persistence | 390 | 400 |
| Historical average | 452 | 459 |
| Ridge regression | 313 | 329 |
| MLP (Keras) | 322 | 340 |
| XGBoost | 292 | 301 |
| **XGBoost + masking augmentation (selected)** | **291** | **301** |

The selected model is also robust to sensor failures (see `A2`/`A3`). Six sequence neural networks
(`A5`) and extra congestion features (`A4`) did not improve on it.

## Repository structure

```
code/
  traffic_prep.py                       cleaning, aggregation, features (shared module)
  A1_allday_cleaning_EDA_features.ipynb cleaning, exploratory analysis, traffic regimes, features, leakage checks
  A2_allday_modelling.ipynb             baselines, ridge, MLP, XGBoost, regime-split XGBoost, augmentation, final evaluation
  A3_allday_statistics.ipynb            model graphs and statistical analysis
  A4_allday_congestion_fixes.ipynb      follow-up: congestion features, 1-min nowcast, bias correction
  A5_allday_neural_networks.ipynb       follow-up: LSTM, GRU, CNNs, ConvLSTM on 60-min sequences
  01_*, 02_*                            earlier study on the morning-only files
  build_notebook*.py                    scripts that generated the notebooks (do not rerun: they overwrite them)
figures_allday/, figures/               all figures
models_allday/, models/                 trained models
results_allday/, results/               result tables (csv)
```

## Models

* Selected model: `models_allday/xgb_single_aug_q_p55620.json` (XGBoost; load with `xgb.XGBRegressor().load_model(...)`,
  inputs = `prep_q_p55620.joblib["x_cols"]`).
* Best Keras model (`.keras` deliverable): `models_allday/nn_best_aug_q_p55620.keras` (LSTM + masking augmentation);
  input preprocessing in `models_allday/nn_config_q_p55620.json`.

## Data

The Trafikverket data are **not included** and may not be shared publicly. To run the code, place the course files in
`dataset_for_traffic_project_assignment_ALLday/` (`training_dataset.csv`, `evaluation_dataset.csv`,
`final_evaluation_dataset.csv`) next to the `code/` folder.

## How to run

1. `pip install -r requirements.txt`
2. Run the notebooks from the `code/` folder in order: **A1 → A2 → A3 → A4 → A5** (Restart + Run All each).
3. A2, A4 and A5 cache trained models in `models_allday/cache*/`; set `REFIT = True` to retrain from scratch
   (the caches are not in this repository, so a fresh clone retrains automatically).

Notes: `statsmodels` must be ≥ 0.14.6 with pandas 3. On Apple-silicon Macs with `tensorflow-metal`, the notebooks
switch TensorFlow to the CPU, because the Metal GPU plugin produced wrong predictions.
