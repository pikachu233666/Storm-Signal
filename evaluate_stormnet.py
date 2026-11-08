"""Evaluate StormNet V2 predictions on identical timestamps.

All generated report text and figures are in English.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    lookup = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Could not find {kind} column. Available columns: {list(df.columns)}")


def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    time_col = find_column(df, ['timestamp_utc', 'datetime_utc', 'timestamp', 'datetime', 'date', 'time'], 'timestamp')
    actual_col = find_column(df, ['actual', 'target', 'y_true', 'true', 'target_sym_h_nT', 'actual_sym_h_nT', 'true_sym_h_nT', 'sym_h_true', 'sym_h_nT'], 'actual')
    pred_col = find_column(df, ['prediction', 'predicted', 'y_pred', 'pred', 'prediction_sym_h_nT', 'predicted_sym_h_nT', 'pred_sym_h_nT', 'forecast'], 'prediction')
    out = pd.DataFrame({
        'timestamp_utc': pd.to_datetime(df[time_col], utc=True, errors='coerce'),
        'actual': pd.to_numeric(df[actual_col], errors='coerce'),
        label: pd.to_numeric(df[pred_col], errors='coerce'),
    })
    for col in ['persistence', 'persistence_prediction', 'persistence_pred', 'current_sym_h_nT', 'current_sym_h', 'last_sym_h_nT', 'baseline_persistence']:
        if col in df.columns:
            out['persistence'] = pd.to_numeric(df[col], errors='coerce')
            break
    out = out.dropna(subset=['timestamp_utc', 'actual', label])
    if out['timestamp_utc'].duplicated().any():
        raise ValueError(f'Duplicate timestamps in {label} predictions')
    return out


def scores(actual: np.ndarray, pred: np.ndarray) -> dict:
    delta = np.asarray(pred, dtype=float) - np.asarray(actual, dtype=float)
    return {'samples': int(len(delta)), 'mae': float(np.mean(np.abs(delta))), 'rmse': float(np.sqrt(np.mean(delta ** 2))), 'bias': float(np.mean(delta))}


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate StormNet baseline and fusion predictions')
    parser.add_argument('--baseline', default='outputs/stormnet_v2_baseline_60m/test_predictions.csv')
    parser.add_argument('--fusion', default='outputs/stormnet_v2_fusion_60m/test_predictions.csv')
    parser.add_argument('--baseline-metrics', default='outputs/stormnet_v2_baseline_60m/metrics.json')
    parser.add_argument('--fusion-metrics', default='outputs/stormnet_v2_fusion_60m/metrics.json')
    parser.add_argument('--output', default='outputs/stormnet_evaluation')
    parser.add_argument('--storm-threshold', type=float, default=-50.0, help='Storm subset: actual SYM-H <= threshold')
    parser.add_argument('--plot-points', type=int, default=240)
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading prediction files...', flush=True)
    baseline = normalize(pd.read_csv(args.baseline), 'baseline')
    fusion = normalize(pd.read_csv(args.fusion), 'fusion')
    if 'persistence' in baseline and 'persistence' in fusion:
        fusion = fusion.drop(columns=['persistence'])
    merged = baseline.merge(fusion, on='timestamp_utc', how='inner', suffixes=('_baseline', '_fusion'), validate='one_to_one')
    if merged.empty:
        raise ValueError('No shared prediction timestamps. Check both CSV files.')
    if not np.allclose(merged['actual_baseline'], merged['actual_fusion'], atol=1e-4, rtol=0):
        raise ValueError('Targets differ between baseline and fusion on shared timestamps.')
    merged = merged.rename(columns={'actual_baseline': 'actual'}).drop(columns=['actual_fusion']).sort_values('timestamp_utc').reset_index(drop=True)
    if 'persistence_baseline' in merged:
        merged = merged.rename(columns={'persistence_baseline': 'persistence'})
    if 'persistence_fusion' in merged:
        merged = merged.rename(columns={'persistence_fusion': 'persistence'})

    expected_counts = {}
    for name, path in [('baseline', args.baseline_metrics), ('fusion', args.fusion_metrics)]:
        p = Path(path)
        if p.exists():
            with p.open(encoding='utf-8') as f:
                expected_counts[name] = json.load(f).get('test', {}).get('samples')
    for name, count in expected_counts.items():
        if count is not None and count != len(merged):
            print(f'WARNING: {name} metrics contain {count} samples but shared CSVs contain {len(merged)}; only shared rows will be evaluated.')

    names = ['baseline', 'fusion']
    if 'persistence' in merged:
        merged = merged.dropna(subset=['persistence'])
        names.insert(0, 'persistence')
    else:
        print('WARNING: No persistence prediction column found; persistence comparison will be omitted.')
    if merged.empty:
        raise ValueError('No valid shared rows remain after cleaning.')
    storm = merged['actual'] <= args.storm_threshold
    results = {
        'shared_samples': int(len(merged)),
        'storm_threshold_nT': args.storm_threshold,
        'storm_samples': int(storm.sum()),
        'all': {name: scores(merged['actual'].to_numpy(), merged[name].to_numpy()) for name in names},
        'storm_only': {name: scores(merged.loc[storm, 'actual'].to_numpy(), merged.loc[storm, name].to_numpy()) for name in names} if storm.any() else {},
    }
    with (out_dir / 'evaluation_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    merged.to_csv(out_dir / 'aligned_test_predictions.csv', index=False)

    rows = []
    for subset in ['all', 'storm_only']:
        for name, vals in results[subset].items():
            rows.append({'subset': subset, 'model': name, **vals})
    pd.DataFrame(rows).to_csv(out_dir / 'model_comparison.csv', index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(names))
    width = 0.36
    ax.bar(xs - width / 2, [results['all'][n]['mae'] for n in names], width, label='MAE')
    ax.bar(xs + width / 2, [results['all'][n]['rmse'] for n in names], width, label='RMSE')
    ax.set_xticks(xs, [n.title() for n in names])
    ax.set_ylabel('Error (nT)')
    ax.set_title('StormNet V2 — Shared Test Set')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'model_comparison.png', dpi=180)
    plt.close(fig)

    if storm.any():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([n.title() for n in names], [results['storm_only'][n]['mae'] for n in names])
        ax.set_ylabel('MAE (nT)')
        ax.set_title(f'Storm Subset — Actual SYM-H <= {args.storm_threshold:g} nT (n={storm.sum()})')
        fig.tight_layout()
        fig.savefig(out_dir / 'storm_comparison.png', dpi=180)
        plt.close(fig)

    # Choose a contiguous-by-row window centered on the strongest observed storm.
    center = int(merged['actual'].idxmin())
    half = max(1, args.plot_points // 2)
    start = max(0, center - half)
    stop = min(len(merged), start + args.plot_points)
    start = max(0, stop - args.plot_points)
    window = merged.iloc[start:stop]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(window['timestamp_utc'], window['actual'], label='Actual SYM-H', linewidth=1.8)
    for name in names:
        ax.plot(window['timestamp_utc'], window[name], label=name.title(), linewidth=1.1, alpha=0.85)
    ax.set_title('Predictions Around the Lowest SYM-H Test Sample')
    ax.set_xlabel('UTC time (sampled observations; gaps may exist)')
    ax.set_ylabel('SYM-H (nT)')
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / 'prediction_timeline.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in names:
        errors = (merged[name] - merged['actual']).to_numpy()
        ax.hist(errors, bins=60, alpha=0.45, label=name.title(), density=True)
    ax.set_title('Prediction Error Distribution')
    ax.set_xlabel('Prediction minus actual (nT)')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'error_distribution.png', dpi=180)
    plt.close(fig)

    print(json.dumps(results, indent=2))
    print(f'\nSaved evaluation artifacts to: {out_dir.resolve()}')
    print('NOTE: The storm subset is threshold-based; it is not an event-level evaluation.')
    if 'persistence' not in names:
        print('NOTE: Use existing metrics.json files for reported persistence metrics, or add a persistence prediction column to the CSVs.')


if __name__ == '__main__':
    main()
