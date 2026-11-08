
from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd


FEATURES = [
    "bt_nT",
    "bz_gsm_nT",
    "by_gsm_nT",
    "speed_km_s",
    "density_cm3",
    "temp_K",
    "pressure_nPa",
    "efield_mV_m",
]

TARGET = "sym_h_nT"
HISTORY_STEPS = 24
HORIZON_STEPS = 12
STEP_MINUTES = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare shared StormNet training samples."
    )

    parser.add_argument(
        "--omni",
        default=(
            "data/raw/omni/"
            "nasa_omni_solar_wind_storms_5min_2010-2026.csv"
        ),
    )

    parser.add_argument(
        "--ionobench",
        default=(
            "data/raw/ionobench/"
            "IonoBench_chronologicalSplit.pickle"
        ),
    )

    parser.add_argument(
        "--output",
        default="data/processed/fusion_60m",
    )

    return parser.parse_args()


def load_omni(path):
    print("Reading NASA OMNI CSV...", flush=True)

    df = pd.read_csv(path)

    required = ["datetime_utc", TARGET] + FEATURES
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing OMNI columns: {missing}")

    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["datetime_utc"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    if df["datetime_utc"].duplicated().any():
        raise ValueError("Duplicate NASA OMNI timestamps found.")

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def load_ionobench(path):
    print("Loading trusted IonoBench pickle...", flush=True)

    # Only unpickle datasets obtained from trusted sources.
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    dates = pd.DatetimeIndex(
        pd.to_datetime(data["Dates"], utc=True)
    )

    if len(dates) != len(data["NormTEC"]):
        raise ValueError("TEC timestamps and maps have different lengths.")

    split_sets = {}

    for name in ("train", "valid", "test"):
        values = data["split_dates"][name]

        split_sets[name] = set(
            pd.to_datetime(values, utc=True)
        )

    return data, dates, split_sets


def prepare_samples(omni, iono, tec_dates, split_sets):
    print("Aligning NASA OMNI and TEC timestamps...", flush=True)

    omni_dates = pd.DatetimeIndex(omni["datetime_utc"])

    # Convert timestamps to the same UTC representation.
    matched_indices = omni_dates.get_indexer(tec_dates)

    feature_values = omni[FEATURES].to_numpy(
        dtype=np.float32
    )

    target_values = omni[TARGET].to_numpy(
        dtype=np.float32
    )

    # Identify rows with at least one invalid input.
    invalid_features = ~np.isfinite(feature_values).all(axis=1)

    # Prefix sums allow fast validation of complete history windows.
    invalid_prefix = np.concatenate(
        ([0], np.cumsum(invalid_features, dtype=np.int64))
    )

    # Detect missing or irregular 5-minute intervals.
    expected_delta = pd.Timedelta(minutes=STEP_MINUTES)

    irregular = np.zeros(len(omni), dtype=np.int64)

    if len(omni) > 1:
        differences = omni_dates[1:] - omni_dates[:-1]
        irregular[1:] = (
            differences != expected_delta
        ).astype(np.int64)

    irregular_prefix = np.concatenate(
        ([0], np.cumsum(irregular, dtype=np.int64))
    )

    split_labels = {}

    for split_name, timestamps in split_sets.items():
        for timestamp in timestamps:
            if timestamp in split_labels:
                raise ValueError(
                    f"Timestamp belongs to multiple splits: {timestamp}"
                )
            split_labels[timestamp] = split_name

    split_ends = {
        name: max(values) if values else None
        for name, values in split_sets.items()
    }

    stats = {
        "tec_total": len(tec_dates),
        "in_split": 0,
        "matched_timestamp": 0,
        "has_history_and_future": 0,
        "valid_target": 0,
        "continuous_history": 0,
        "valid_features": 0,
        "valid_tec": 0,
        "accepted": 0,
    }

    samples = []

    for tec_idx, (timestamp, omni_idx) in enumerate(
        zip(tec_dates, matched_indices)
    ):
        split_name = split_labels.get(timestamp)

        if split_name is None:
            continue

        stats["in_split"] += 1

        if omni_idx < 0:
            continue

        stats["matched_timestamp"] += 1

        history_start = omni_idx - HISTORY_STEPS + 1
        target_idx = omni_idx + HORIZON_STEPS

        if history_start < 0 or target_idx >= len(omni):
            continue

        # Prevent labels from crossing the end of a split.
        split_end = split_ends[split_name]

        if omni_dates[target_idx] > split_end:
            continue

        # Verify the target is exactly 60 minutes ahead.
        target_delta = (
            omni_dates[target_idx] - timestamp
        )

        if target_delta != pd.Timedelta(minutes=60):
            continue

        stats["has_history_and_future"] += 1

        target = target_values[target_idx]

        if not np.isfinite(target):
            continue

        stats["valid_target"] += 1

        # Verify the historical window is exactly 24 x 5 minutes.
        if (
            omni_dates[omni_idx] - omni_dates[history_start]
            != pd.Timedelta(minutes=(HISTORY_STEPS - 1) * 5)
        ):
            continue

        irregular_count = (
            irregular_prefix[omni_idx + 1]
            - irregular_prefix[history_start + 1]
        )

        if irregular_count != 0:
            continue

        stats["continuous_history"] += 1

        invalid_count = (
            invalid_prefix[omni_idx + 1]
            - invalid_prefix[history_start]
        )

        if invalid_count != 0:
            continue

        stats["valid_features"] += 1

        tec_map = iono["NormTEC"][tec_idx]

        if not np.isfinite(tec_map).all():
            continue

        stats["valid_tec"] += 1

        samples.append({
            "timestamp_utc": timestamp.isoformat(),
            "split": split_name,
            "omni_start_idx": int(history_start),
            "omni_end_idx": int(omni_idx),
            "tec_idx": int(tec_idx),
            "target_idx": int(target_idx),
            "target_sym_h_nT": float(target),
        })

        stats["accepted"] += 1

    return pd.DataFrame(samples), stats


def main():
    args = parse_args()

    omni_path = Path(args.omni)
    iono_path = Path(args.ionobench)
    output_dir = Path(args.output)

    if not omni_path.exists():
        raise FileNotFoundError(omni_path)

    if not iono_path.exists():
        raise FileNotFoundError(iono_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    omni = load_omni(omni_path)
    iono, tec_dates, split_sets = load_ionobench(iono_path)

    print("\n=== Source Information ===")
    print("NASA OMNI rows:", len(omni))
    print("TEC maps:", len(tec_dates))
    print("OMNI start:", omni["datetime_utc"].min())
    print("OMNI end:", omni["datetime_utc"].max())
    print("TEC start:", tec_dates.min())
    print("TEC end:", tec_dates.max())

    samples, stats = prepare_samples(
        omni,
        iono,
        tec_dates,
        split_sets,
    )

    print("\n=== Alignment Diagnostics ===")

    for name, value in stats.items():
        print(f"{name}: {value}")

    if samples.empty:
        raise RuntimeError(
            "No aligned samples found. See diagnostics above."
        )

    samples_path = output_dir / "shared_samples.csv"
    samples.to_csv(samples_path, index=False)

    metadata = {
        "dataset": "Storm Signal Shared Dataset",
        "omni_path": str(omni_path.resolve()),
        "ionobench_path": str(iono_path.resolve()),
        "features": FEATURES,
        "target": TARGET,
        "history_steps": HISTORY_STEPS,
        "history_minutes": HISTORY_STEPS * STEP_MINUTES,
        "horizon_steps": HORIZON_STEPS,
        "horizon_minutes": HORIZON_STEPS * STEP_MINUTES,
        "tec_shape": list(iono["NormTEC"].shape[1:]),
        "sample_counts": {
            str(k): int(v)
            for k, v in samples["split"].value_counts().items()
        },
        "diagnostics": stats,
    }

    metadata_path = output_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\n=== Dataset Summary ===")

    for name in ("train", "valid", "test"):
        count = int((samples["split"] == name).sum())
        print(f"{name}: {count:,} samples")

    print("\nSaved:", samples_path)
    print("Saved:", metadata_path)
    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()
