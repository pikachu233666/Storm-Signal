"""Train a leakage-aware 60-minute NASA OMNI SYM-H LightGBM baseline."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["bt_nT", "bz_gsm_nT", "by_gsm_nT", "speed_km_s", "density_cm3", "pressure_nPa", "sym_h_nT"]
LAGS = [0, 1, 3, 6, 12, 24]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/baseline")
    parser.add_argument("--horizon", type=int, default=12, help="Number of 5-minute steps")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be >= 1")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, usecols=["datetime_utc", *FEATURES], low_memory=False)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc").drop_duplicates("datetime_utc", keep="last")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Reindex to a regular time grid so missing intervals never become fake 5-minute lags.
    df = df.set_index("datetime_utc").asfreq("5min")
    # Never interpolate across unknown future values. Missing predictors remain NaN for LightGBM.
    cols = []
    for col in FEATURES:
        for lag in LAGS:
            name = f"{col}_lag{lag}"
            df[name] = df[col].shift(lag)
            cols.append(name)
    target = "target_sym_h"
    df[target] = df["sym_h_nT"].shift(-args.horizon)
    df = df.dropna(subset=[target, "sym_h_nT"])
    # Split by target timestamp, not feature timestamp, to prevent cross-boundary labels.
    target_times = df.index + pd.Timedelta(minutes=5 * args.horizon)
    train = df.loc[target_times < pd.Timestamp("2023-01-01", tz="UTC")]
    valid = df.loc[(target_times >= pd.Timestamp("2023-01-01", tz="UTC")) & (target_times < pd.Timestamp("2025-01-01", tz="UTC"))]
    test = df.loc[target_times >= pd.Timestamp("2025-01-01", tz="UTC")]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"Empty split: train={len(train)}, valid={len(valid)}, test={len(test)}")
    model = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.035, num_leaves=31, max_depth=-1, random_state=42, verbosity=-1, n_jobs=-1)
    model.fit(train[cols], train[target], eval_set=[(valid[cols], valid[target])], eval_metric="l1", callbacks=[lgb.early_stopping(60, verbose=False)])
    pred = model.predict(test[cols])
    actual = test[target].to_numpy()
    persistence = test["sym_h_nT"].to_numpy()
    metrics = {
        "horizon_minutes": args.horizon * 5,
        "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "persistence_mae": float(mean_absolute_error(actual, persistence)),
        "persistence_rmse": float(np.sqrt(mean_squared_error(actual, persistence))),
        "best_iteration": int(model.best_iteration_),
        "test_start": str(test.index.min()), "test_end": str(test.index.max()),
    }
    model.booster_.save_model(str(out / "lightgbm.txt"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"datetime_utc": test.index.astype(str), "actual_sym_h": actual, "predicted_sym_h": pred, "persistence_sym_h": persistence}).to_csv(out / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {out.resolve()}")

if __name__ == "__main__":
    main()
