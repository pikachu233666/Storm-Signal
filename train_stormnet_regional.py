#!/usr/bin/env python3
"""Storm Signal regional TEC-change forecasting, 2-hour horizon.

Uses existing IonoBench and NASA OMNI files. Output probabilities describe
large TEC changes, NOT local geomagnetic storms or grid outages.
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

FEATURES = ['bt_nT', 'bz_gsm_nT', 'by_gsm_nT', 'speed_km_s', 'density_cm3', 'temp_K', 'pressure_nPa', 'efield_mV_m']


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--iono', default='data/raw/ionobench/IonoBench_chronologicalSplit.pickle')
    p.add_argument('--omni', default='data/raw/omni/nasa_omni_solar_wind_storms_5min_2010-2026.csv')
    p.add_argument('--output', default='outputs/stormnet_regional_2h')
    p.add_argument('--shared', default='data/processed/fusion_60m/shared_samples.csv')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threshold-quantile', type=float, default=0.90)
    p.add_argument('--patience', type=int, default=6)
    return p.parse_args()


def as_utc(values):
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def build_data(a):
    print('Loading trusted IonoBench pickle...', flush=True)
    with open(a.iono, 'rb') as f:
        raw = pickle.load(f)  # Only load pickle files from trusted sources.
    tec = np.asarray(raw['NormTEC'], dtype=np.float32)
    dates = as_utc(raw['Dates'])
    if tec.ndim != 3 or tec.shape[1:] != (72, 72) or len(dates) != len(tec):
        raise ValueError('Unexpected IonoBench TEC shape or timestamps')
    print('TEC timestamp dtype:', dates.dtype, flush=True)
    print('Loading OMNI...', flush=True)
    df = pd.read_csv(a.omni, usecols=['datetime_utc'] + FEATURES)
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    df = df.sort_values('datetime_utc').drop_duplicates('datetime_utc').set_index('datetime_utc')
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='5min', tz='UTC'))
    omni = df[FEATURES].to_numpy(dtype=np.float32)
    # Reuse the already validated OMNI/TEC anchor alignment from fusion preprocessing.
    shared_path = Path(a.shared)
    if not shared_path.is_file():
        raise FileNotFoundError(f'Missing validated alignment file: {shared_path}')
    shared = pd.read_csv(shared_path)
    required = {'timestamp_utc', 'split', 'omni_end_idx', 'tec_idx'}
    if not required.issubset(shared.columns):
        raise ValueError(f'Shared CSV missing columns: {sorted(required - set(shared.columns))}')
    shared['timestamp_utc'] = pd.to_datetime(shared['timestamp_utc'], utc=True)
    date_pos = {ts: i for i, ts in enumerate(dates)}
    omni_pos = pd.Series(np.arange(len(omni), dtype=np.int64), index=df.index)
    records = {'train': [], 'valid': [], 'test': []}
    # Use IonoBench's original split membership for the future TEC label.
    split_dates = raw['split_dates']
    memberships = {split: set(as_utc(split_dates[split]))
                   for split in records}
    for split in records:
        stats = {'anchors': 0, 'split_future': 0, 'missing_future': 0,
                 'anchor_mismatch': 0, 'omni_missing': 0, 'nonfinite': 0}
        subset = shared.loc[shared['split'] == split]
        for row in subset.itertuples(index=False):
            stats['anchors'] += 1
            ts = row.timestamp_utc
            i, k = int(row.tec_idx), int(row.omni_end_idx)
            if i < 0 or i >= len(tec) or k < 23 or k >= len(omni):
                stats['anchor_mismatch'] += 1
                continue
            if dates[i] != ts or df.index[k] != ts:
                # Protect against mismatched CSV indexing / alternate OMNI reindexing.
                i = date_pos.get(ts, -1)
                k_val = omni_pos.get(ts, np.nan)
                if i < 0 or pd.isna(k_val):
                    stats['anchor_mismatch'] += 1
                    continue
                k = int(k_val)
                if k < 23:
                    stats['anchor_mismatch'] += 1
                    continue
            future = ts + pd.Timedelta(hours=2)
            j = date_pos.get(future)
            if j is None:
                stats['missing_future'] += 1
                continue
            if future not in memberships[split]:
                stats['split_future'] += 1
                continue
            hist = omni[k-23:k+1]
            if len(hist) != 24:
                stats['omni_missing'] += 1
                continue
            if not (np.isfinite(hist).all() and np.isfinite(tec[i]).all()
                    and np.isfinite(tec[j]).all()):
                stats['nonfinite'] += 1
                continue
            records[split].append((i, j, k, ts.isoformat()))
        print(f'{split}: {len(records[split])} usable paired samples; diagnostics={stats}', flush=True)
        if not records[split]:
            raise ValueError(f'No usable samples in {split}. Inspect diagnostics above; '
                             'the CSV and IonoBench split timestamps may differ.')
    # Estimate feature scaling from training history only.
    train_hist = np.stack([omni[k-23:k+1] for _, _, k, _ in records['train']])
    mean = train_hist.mean(axis=(0, 1)).astype(np.float32)
    std = train_hist.std(axis=(0, 1)).astype(np.float32).clip(min=1e-5)
    del train_hist
    # Define a global, pre-registered event threshold using train data only.
    # A sample is positive where absolute normalized TEC changes exceed threshold.
    # Sample a deterministic subset of cells to avoid allocating a huge tensor.
    rng = np.random.default_rng(a.seed)
    values = []
    for i, j, _, _ in records['train']:
        picks = rng.integers(0, 72 * 72, size=256)
        delta = np.abs(tec[j].ravel()[picks] - tec[i].ravel()[picks])
        values.append(delta)
    threshold = float(np.quantile(np.concatenate(values), a.threshold_quantile))
    del values
    print(f'Train-only absolute TEC-change threshold: {threshold:.6f}', flush=True)
    return tec, omni, records, mean, std, threshold


class RegionalDataset(Dataset):
    def __init__(self, tec, omni, records, mean, std, threshold):
        self.tec, self.omni, self.records = tec, omni, records
        self.mean, self.std, self.threshold = mean, std, threshold

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        i, j, k, _ = self.records[index]
        history = ((self.omni[k-23:k+1] - self.mean) / self.std).astype(np.float32)
        current = self.tec[i][None, :, :]
        target = (np.abs(self.tec[j] - self.tec[i]) >= self.threshold).astype(np.float32)[None, :, :]
        return torch.from_numpy(history), torch.from_numpy(current.copy()), torch.from_numpy(target)


class RegionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(8, 48, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(size=(36, 36), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(size=(72, 72), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, history, current):
        _, hidden = self.gru(history)
        spatial = self.encoder(current)
        condition = hidden[-1, :, :, None, None].expand(-1, -1, spatial.shape[-2], spatial.shape[-1])
        return self.decoder(torch.cat([spatial, condition], dim=1))


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, save=False):
    model.eval()
    losses, n = 0.0, 0
    brier_sum, abs_sum, total, positives = 0.0, 0.0, 0, 0
    maps, labels = [], []
    for hist, current, target in loader:
        hist, current, target = hist.to(device), current.to(device), target.to(device)
        logits = model(hist, current)
        loss = loss_fn(logits, target)
        losses += loss.item() * len(hist)
        n += len(hist)
        prob = torch.sigmoid(logits)
        brier_sum += ((prob - target) ** 2).sum().item()
        abs_sum += (prob - target).abs().sum().item()
        total += target.numel()
        positives += target.sum().item()
        if save:
            maps.append(prob.cpu().numpy().astype(np.float16))
            labels.append(target.cpu().numpy().astype(np.uint8))
    metrics = {'bce': losses / n, 'brier': brier_sum / total,
               'probability_mae': abs_sum / total, 'positive_fraction': positives / total,
               'samples': n}
    return metrics, (np.concatenate(maps) if save else None), (np.concatenate(labels) if save else None)


def main():
    a = args_parser()
    if not 0 < a.threshold_quantile < 1:
        raise ValueError('threshold-quantile must be between 0 and 1')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    tec, omni, records, mean, std, threshold = build_data(a)
    datasets = {s: RegionalDataset(tec, omni, records[s], mean, std, threshold) for s in records}
    loaders = {s: DataLoader(datasets[s], batch_size=a.batch_size, shuffle=(s == 'train'),
                             num_workers=a.workers, pin_memory=(device.type == 'cuda')) for s in datasets}
    model = RegionalNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best, stale = float('inf'), 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for hist, current, target in loaders['train']:
            hist, current, target = hist.to(device), current.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(hist, current)
            loss = loss_fn(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(hist)
            total_n += len(hist)
        valid, _, _ = evaluate(model, loaders['valid'], device, loss_fn)
        print(f'Epoch {epoch:02d}: train_bce={total_loss / total_n:.5f} '
              f'valid_bce={valid["bce"]:.5f} valid_brier={valid["brier"]:.5f}', flush=True)
        if valid['bce'] < best - 1e-5:
            best, stale = valid['bce'], 0
            torch.save(model.state_dict(), out / 'best_model.pt')
        else:
            stale += 1
            if stale >= a.patience:
                print('Early stopping.', flush=True)
                break
    model.load_state_dict(torch.load(out / 'best_model.pt', map_location=device, weights_only=True))
    valid, _, _ = evaluate(model, loaders['valid'], device, loss_fn)
    test, predictions, targets = evaluate(model, loaders['test'], device, loss_fn, save=True)
    np.savez_compressed(out / 'test_probability_maps.npz', probabilities=predictions, labels=targets,
                        timestamps=np.asarray([r[3] for r in records['test']]))
    metrics = {'model': 'regional_tec_change_cnn_gru', 'horizon_hours': 2,
               'target': 'absolute change in IonoBench normalized TEC, per grid cell',
               'threshold_quantile_train': a.threshold_quantile, 'threshold': threshold,
                'threshold_estimation': '256 random TEC grid cells per training sample',
               'valid': valid, 'test': test,
               'caveat': 'Probabilities are uncalibrated. Not geomagnetic storm or power outage probabilities.'}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (out / 'normalization.json').write_text(json.dumps({'features': FEATURES, 'mean': mean.tolist(),
                                                        'std': std.tolist(), 'threshold': threshold}, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)
    print('Saved:', out.resolve(), flush=True)


if __name__ == '__main__':
    main()
