"""Train residual StormNet v2 baseline or TEC fusion on shared samples."""
from __future__ import annotations

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


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['baseline', 'fusion'], required=True)
    p.add_argument('--samples', default='data/processed/fusion_60m/shared_samples.csv')
    p.add_argument('--metadata', default='data/processed/fusion_60m/metadata.json')
    p.add_argument('--omni', default=None, help='Override OMNI path in metadata')
    p.add_argument('--ionobench', default=None, help='Override IonoBench path in metadata')
    p.add_argument('--output', default=None)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--storm-threshold', type=float, default=-50.0)
    return p.parse_args()


class StormDataset(Dataset):
    def __init__(self, frame, features, tec, mean, std, history_steps, current_sym_h, current_mean, current_std):
        self.frame = frame.reset_index(drop=True)
        self.features = features
        self.tec = tec
        self.mean = mean
        self.std = std
        self.history_steps = history_steps
        self.current_sym_h = current_sym_h
        self.current_mean = current_mean
        self.current_std = current_std
        self.starts = self.frame['omni_start_idx'].to_numpy(dtype=np.int64)
        self.ends = self.frame['omni_end_idx'].to_numpy(dtype=np.int64)
        self.tec_ids = self.frame['tec_idx'].to_numpy(dtype=np.int64)
        self.targets = self.frame['target_sym_h_nT'].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = self.ends[idx]
        x = self.features[start:end + 1]
        if len(x) != self.history_steps:
            raise ValueError('Unexpected history length')
        x = (x - self.mean) / self.std
        current = np.float32(self.current_sym_h[end])
        current_feature = np.full((len(x), 1), (current - self.current_mean) / self.current_std, dtype=np.float32)
        x = np.concatenate([x, current_feature], axis=1)
        if self.tec is None:
            return torch.from_numpy(x.astype(np.float32)), torch.tensor(current), torch.tensor(self.targets[idx])
        image = np.asarray(self.tec[self.tec_ids[idx]], dtype=np.float32)
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(image.copy()).unsqueeze(0),
            torch.tensor(current),
            torch.tensor(self.targets[idx]),
        )


class StormNet(nn.Module):
    def __init__(self, n_features, fusion=False):
        super().__init__()
        self.fusion = fusion
        self.gru = nn.GRU(n_features, 64, batch_first=True)
        if fusion:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            )
        self.head = nn.Sequential(
            nn.Linear(64 + (32 if fusion else 0), 64),
            nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x, tec=None):
        _, h = self.gru(x)
        z = h[-1]
        if self.fusion:
            if tec is None:
                raise ValueError('TEC input required for fusion')
            z = torch.cat([z, self.cnn(tec)], dim=1)
        return self.head(z).squeeze(1)


def run_epoch(model, loader, optimizer, device, scaler, use_amp, delta_std):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    n = 0
    for batch in loader:
        if len(batch) == 3:
            x, current, y = batch
            tec = None
        else:
            x, tec, current, y = batch
            tec = tec.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        current = current.to(device, non_blocking=True)
        y_scaled = (y - current) / delta_std
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(x, tec)
                loss = nn.functional.smooth_l1_loss(prediction, y_scaled)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        loss_sum += loss.item() * len(y)
        n += len(y)
    return loss_sum / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, use_amp, delta_std):
    model.eval()
    predictions = []
    targets = []
    for batch in loader:
        if len(batch) == 3:
            x, current, y = batch
            tec = None
        else:
            x, tec, current, y = batch
            tec = tec.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(x.to(device), tec)
        predictions.append(current.numpy() + output.float().cpu().numpy() * delta_std)
        targets.append(y.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def metrics(pred, true):
    return {
        'mae': float(np.mean(np.abs(pred - true))),
        'rmse': float(np.sqrt(np.mean((pred - true) ** 2))),
        'samples': int(len(true)),
    }


def main():
    args = arguments()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError('epochs, batch-size, and patience must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'Device: {device}', flush=True)

    with open(args.metadata, encoding='utf-8') as f:
        meta = json.load(f)
    samples = pd.read_csv(args.samples)
    for split in ('train', 'valid', 'test'):
        if not (samples['split'] == split).any():
            raise ValueError(f'No samples for split {split}')
    features = meta['features']
    history = int(meta['history_steps'])
    omni_path = args.omni or meta['omni_path']
    iono_path = args.ionobench or meta['ionobench_path']
    print('Reading OMNI...', flush=True)
    omni = pd.read_csv(omni_path, usecols=features + ['sym_h_nT'])
    values = omni[features].to_numpy(dtype=np.float32)
    current_sym_h = omni['sym_h_nT'].to_numpy(dtype=np.float32)
    if samples['omni_end_idx'].max() >= len(values):
        raise ValueError('Sample indices exceed OMNI file; check original row order')
    current_for_samples = current_sym_h[samples['omni_end_idx'].to_numpy(dtype=np.int64)]
    valid_current = np.isfinite(current_for_samples)
    if not valid_current.all():
        print(f"Excluding {int((~valid_current).sum())} samples with missing current SYM-H", flush=True)
        samples = samples.loc[valid_current].reset_index(drop=True)
    for split in ('train', 'valid', 'test'):
        if not (samples['split'] == split).any():
            raise ValueError(f'No usable samples for split {split}')

    # Fit input scaling only on training histories, not validation or test.
    train = samples.loc[samples['split'] == 'train'].reset_index(drop=True)
    used = np.zeros(len(values) + 1, dtype=np.int64)
    for start, end in zip(train['omni_start_idx'], train['omni_end_idx']):
        used[int(start)] += 1
        used[int(end) + 1] -= 1
    used = np.cumsum(used[:-1]) > 0
    train_values = values[used]
    mean = np.mean(train_values, axis=0).astype(np.float32)
    std = np.std(train_values, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    train_current = current_sym_h[train['omni_end_idx'].to_numpy(dtype=np.int64)]
    train_delta = train['target_sym_h_nT'].to_numpy(dtype=np.float32) - train_current
    delta_std = max(float(np.std(train_delta)), 1e-6)
    current_mean = float(np.mean(train_current))
    current_std = max(float(np.std(train_current)), 1e-6)
    print(f'Training delta std: {delta_std:.4f} nT', flush=True)

    tec = None
    if args.model == 'fusion':
        print('Loading trusted IonoBench pickle (may take a while)...', flush=True)
        with open(iono_path, 'rb') as f:
            iono = pickle.load(f)  # Trusted source only; pickle executes code.
        tec = iono['NormTEC']
        if tuple(tec.shape[1:]) != (72, 72):
            raise ValueError(f'Unexpected TEC map shape: {tec.shape}')
        if samples['tec_idx'].max() >= len(tec):
            raise ValueError('TEC indices exceed dataset')
        print('Note: TEC normalization was provided by IonoBench; audit its fitting method.', flush=True)

    datasets = {
        split: StormDataset(
            samples.loc[samples['split'] == split], values, tec, mean, std, history, current_sym_h, current_mean, current_std
        )
        for split in ('train', 'valid', 'test')
    }
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(split == 'train'),
            num_workers=0, pin_memory=use_amp,
        )
        for split, dataset in datasets.items()
    }
    model = StormNet(len(features) + 1, fusion=args.model == 'fusion').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    output = Path(args.output or f'outputs/stormnet_v2_{args.model}_60m')
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    wait = 0
    checkpoint = output / 'best_model.pt'
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders['train'], optimizer, device, scaler, use_amp, delta_std)
        valid_loss = run_epoch(model, loaders['valid'], None, device, scaler, use_amp, delta_std)
        print(f'Epoch {epoch:02d}: train_loss={train_loss:.5f} valid_loss={valid_loss:.5f}', flush=True)
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            wait = 0
            torch.save({
                'state_dict': model.state_dict(), 'features': features,
                'input_mean': mean.tolist(), 'input_std': std.tolist(),
                'delta_std': delta_std, 'target_mode': 'residual',
                'current_mean': current_mean, 'current_std': current_std,
                'model': args.model, 'history_steps': history,
            }, checkpoint)
        else:
            wait += 1
            if wait >= args.patience:
                print('Early stopping.', flush=True)
                break

    # The checkpoint is created by this script, not from an untrusted source.
    # Only state_dict tensors and primitive Python types are saved.
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['state_dict'])
    result = {'model': args.model, 'version': 'v2_residual', 'device': str(device), 'best_validation_loss': best_loss, 'delta_std': delta_std}
    for split in ('valid', 'test'):
        frame = samples.loc[samples['split'] == split].reset_index(drop=True)
        pred, true = predict(model, loaders[split], device, use_amp, delta_std)
        baseline = current_sym_h[frame['omni_end_idx'].to_numpy(dtype=np.int64)]
        result[split] = metrics(pred, true)
        result[split]['persistence'] = metrics(baseline, true)
        storm_mask = true <= args.storm_threshold
        if storm_mask.any():
            result[split]['storm_only'] = metrics(pred[storm_mask], true[storm_mask])
            result[split]['storm_only']['persistence'] = metrics(baseline[storm_mask], true[storm_mask])
        else:
            result[split]['storm_only'] = {'samples': 0}
        pd.DataFrame({
            'timestamp_utc': frame['timestamp_utc'], 'actual_sym_h_nT': true,
            'predicted_sym_h_nT': pred, 'persistence_sym_h_nT': baseline,
        }).to_csv(output / f'{split}_predictions.csv', index=False)
    with open(output / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f'Saved outputs to: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
