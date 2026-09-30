# Machine Learning

Two independent models, each with its own dataset, notebooks and exported
artifacts. The backend loads the exported `.pkl` files at startup and never
touches the notebooks.

```
ml/
├── occupancy/            How full the bus is — classification
│   ├── notebooks/        Download, feature engineering, training
│   ├── models/           Artifacts the backend loads
│   ├── data/             Encoders (datasets themselves are not committed)
│   └── scripts/          One-off helpers
├── eta/                  Minutes to the next stop — regression
│   ├── notebooks/        Training and evaluation, plus earlier experiments
│   └── model/            Artifacts the backend loads
└── requirements.txt      Dependencies for the notebooks
```

Model files are stored with Git LFS. Run `git lfs install` before cloning, or
`git lfs pull` afterwards, otherwise the `.pkl` files arrive as text pointers
and the backend starts in fallback mode.

Datasets are deliberately not committed — they are tens of millions of rows.
The notebooks download or read them from a local path, and `.gitignore`
excludes `.csv` and `.parquet` under both data folders.

## Occupancy model

Predicts how full a bus is as one of four classes.

**Dataset.** SUNT OD, the public origin-destination dataset for Salvador,
Brazil, published on Hugging Face as `labiaufba/PublicTransportationSunt`. The
committed model was trained on March 2024: 18,291,730 stop-level records after
cleaning, split 80/20 into 14,633,384 training and 3,658,346 test records.

**Target.** The raw `loading` column (passengers on board) discretized against
an assumed capacity of 80 passengers:

| Class | Share of capacity | Passengers | Share of dataset |
|---|---|---|---|
| low | 0–24% | 0–20 | 58.98% |
| medium | 25–49% | 21–39 | 26.48% |
| high | 50–81% | 40–65 | 11.28% |
| very_high | 82%+ | 66+ | 3.26% |

The classes are imbalanced 18:1 between low and very_high, so training uses
per-sample weights derived from the inverse class frequency.

**Features (14).** `route_short_name`, `direction_id`, `pt_sequence`,
`stop_id`, `hour`, `day_of_week`, `is_weekend`, `is_rush_hour`,
`route_progression`, `loading_lag_1`, `loading_lag_2`, `trip_stage`,
`time_of_day`, `loading_mean_route`.

The lag features are the occupancy at the previous one and two stops of the
same trip. They are computed before any sampling, so the stop sequence stays
intact, and they use only past information — no leakage. The `loading` column
itself and everything derived from it are dropped from X.

**Results.** XGBoost, 300 estimators, max depth 8, learning rate 0.05, early
stopping at 20 rounds. Training took about 6 minutes.

| Metric | Train | Test |
|---|---|---|
| Accuracy | — | 0.9402 |
| Balanced accuracy | 0.9280 | 0.9268 |
| Macro F1 | 0.9169 | 0.9158 |

The 0.12% train-test gap indicates the model generalizes rather than memorizes.
Per-class accuracy on the test set: low 96.45%, very_high 94.20%, medium
90.42%, high 89.66%. The two classes that get confused are high and medium,
which are adjacent and share a boundary at 50% of capacity.

Compared against a majority-class baseline (0.5898 accuracy) and a Random
Forest trained on the same data (0.9413 accuracy, 0.9248 balanced accuracy,
0.9175 macro F1). XGBoost was chosen for slightly better balanced accuracy and
a much smaller artifact.

**Feature importance.** The two lag features dominate:

| Feature | Importance |
|---|---|
| loading_lag_1 | 74.65% |
| loading_lag_2 | 18.92% |
| trip_stage | 1.14% |
| route_progression | 1.13% |
| hour | 0.68% |
| everything else | under 0.7% each |

This is intuitive — the best predictor of how full a bus is right now is how
full it was at the previous stop — but it has a direct consequence for serving.
The backend has to reproduce that history live, which it does by keeping a
two-entry deque of recent passenger counts per bus. See
[../backend/README.md](../backend/README.md).

**Running the notebooks.** In order, from `ml/occupancy/notebooks/`:

1. `01_dataloader_SUNT_OD.ipynb` — downloads the chosen months from Hugging
   Face, validates the critical columns, sorts by route, direction, trip and
   stop sequence, and saves a parquet
2. `02_featureengineering.ipynb` — builds the features and the target, and
   saves `X.parquet`, `y.pkl` and the label encoders
3. `03_trainingRandomForest.ipynb` — the Random Forest comparison
4. `04_trainingXGBoost.ipynb` — trains the model the backend uses and exports
   the artifacts

Install `requirements.txt` first. Step 2 needs the output of step 1, and step 4
reads the Random Forest results from step 3 for the comparison table.

**Artifacts.**

| File | Contents |
|---|---|
| `models/xgboost_occupancy_with_lags.pkl` | The trained classifier |
| `models/xgboost_label_encoder_with_lags.pkl` | Maps predicted integers back to class names |
| `models/xgboost_feature_names_with_lags.pkl` | Feature order — the backend reindexes to match |
| `data/sunt_2024_03_with_lags_encoders.pkl` | Label encoders for the categorical features |
| `data/route_loading_mean.pkl` | Per-route mean loading — not committed, see below |

`route_loading_mean.pkl` is a lookup table for the `loading_mean_route` feature,
so the backend does not need the full training set at inference time. It is
built from the training parquet, which is not in the repository, so a fresh
clone will not have it and the backend falls back to zero for that feature. If
you have the dataset, rebuild it:

```bash
python ml/occupancy/scripts/build_route_loading_mean_lookup.py
```

**Other scripts.** `notebooks/plot_feature_importance.py` regenerates the
feature importance chart from the saved model without retraining anything.

## ETA model

Predicts the minutes until a bus reaches its next stop.

**Dataset.** MTA bus telemetry from New York, June 2017 (`mta_1706.csv`). The
first 3,000,000 records in chronological order were taken as a continuous
block, cleaned down to 2,634,832 usable rows, and split chronologically —
2,107,865 for training (the past) and 526,967 for testing (the future). The
split is by time rather than random, so the model is evaluated on data it could
not have seen, which is the honest setup for a forecasting problem.

**Target.** `TimeToArrival`, the difference between the expected arrival time
and the recording time, in minutes, clipped to the range 0 to 60.

**Features (12).** `DistanceFromStop`, `dist_to_dest_m`, `distance_close`,
`schedule_delay`, `speed_kmh`, `speed_roll3`, `direction`, `hour_sin`,
`hour_cos`, `is_am_rush`, `is_pm_rush`, `proximity_enc`.

Hour of day is encoded as a sine and cosine pair so that 23:00 and 00:00 are
adjacent rather than maximally distant. Speeds are computed separately on the
train and test sets so no future information leaks backwards.

**Results.** XGBoost with hyperparameters from a randomized search over 30
candidates and 3-fold cross-validation.

| Metric | Value |
|---|---|
| MAE | 0.3009 min (18.1 s) |
| RMSE | 0.5172 min |
| R² | 0.9189 |

Best parameters: 703 estimators, max depth 9, learning rate 0.034, subsample
0.863, colsample_bytree 0.998, gamma 0.088, min_child_weight 5, reg_alpha 1.0,
reg_lambda 10.0.

Error is stable across the week — mean absolute error ranges from 0.26 minutes
on Sundays to 0.35 on Wednesdays — and lowest overnight, when traffic is
predictable. The per-hour and per-day breakdowns are in `mae_por_hora.csv` and
`mae_por_dia.csv`.

**Running the notebook.** `notebooks/XGBoost_For_ETA_FINAL.ipynb` is the one
that produced the deployed model. Update `path_csv` at the top to point at your
local copy of the MTA dataset and run it end to end; it handles cleaning,
feature engineering, the chronological split, the search, evaluation and export.

The other notebooks are earlier iterations kept for the record: three Random
Forest versions, two earlier XGBoost versions, two versions reworked for the
academic write-up, and a CNN experiment. None of them produce artifacts the
backend uses. `Random Forest For ETA` at the root of `ml/` is an early notebook
saved without its `.ipynb` extension.

**Artifacts.**

| File | Contents |
|---|---|
| `model/xgb_model_eta_final.pkl` | The trained regressor, loaded with joblib |
| `model/xgb_features_eta_final.pkl` | Feature order and the best parameters |

**The caveat that matters.** This model was trained on New York and is used in
Asunción. Its features are deliberately generic — distance, speed, time of day —
which is why the transfer works at all, and predictions behave sensibly in
practice. But the absolute times encode New York traffic, stop spacing and
schedule adherence. Retraining on locally collected data is the single largest
improvement available to this project.
