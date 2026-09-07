"""
Builds a route_short_name → mean loading lookup table for the occupancy
XGBoost model's `loading_mean_route` feature, so the backend can serve it
at inference time without needing the full training dataset.

Run once (and whenever the occupancy model/encoders are retrained):
    python ml/occupancy/scripts/build_route_loading_mean_lookup.py
"""

import pickle
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent  # ml/occupancy

X = pd.read_parquet(
    BASE / "data" / "sunt_2024_03_with_lags_X.parquet",
    columns=["route_short_name", "loading_mean_route"],
)

nunique_per_route = X.groupby("route_short_name")["loading_mean_route"].nunique()
assert (nunique_per_route <= 1).all(), "loading_mean_route is not constant within a route group"

lookup = X.groupby("route_short_name")["loading_mean_route"].first().to_dict()

out_path = BASE / "data" / "route_loading_mean.pkl"
with open(out_path, "wb") as f:
    pickle.dump(lookup, f)

print(f"Wrote {len(lookup)} routes to {out_path}")
