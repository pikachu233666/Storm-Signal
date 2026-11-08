"""Storm Signal historical replay API. All responses use real saved model outputs."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(os.environ.get('STORM_SIGNAL_ROOT', Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT / 'outputs'
BASE = OUT / 'stormnet_v2_baseline_60m'
FUSION = OUT / 'stormnet_v2_fusion_60m'
REGION = OUT / 'stormnet_regional_2h'

app = FastAPI(title='Storm Signal API', version='1.0.0', description='Historical test-set replay, not live forecasts')
origins = [x.strip() for x in os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=['GET'], allow_headers=['*'])


def read_json(path: Path):
    if not path.is_file():
        raise HTTPException(503, f'Missing {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}. Run training first.')
    return json.loads(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=4)
def read_predictions(model: str):
    path = (BASE if model == 'baseline' else FUSION) / 'test_predictions.csv'
    if not path.is_file():
        raise HTTPException(503, f'Missing {path.name} for {model}')
    df = pd.read_csv(path)
    required = {'timestamp_utc', 'actual_sym_h_nT', 'predicted_sym_h_nT', 'persistence_sym_h_nT'}
    if not required.issubset(df.columns):
        raise HTTPException(500, f'Unexpected columns in {path.name}: {list(df.columns)}')
    return df


@lru_cache(maxsize=1)
def read_maps():
    path = REGION / 'test_probability_maps.npz'
    if not path.is_file():
        raise HTTPException(503, 'Missing regional test_probability_maps.npz. Train regional model first.')
    with np.load(path, allow_pickle=False) as data:
        probabilities = data['probabilities']
        timestamps = data['timestamps'].astype(str)
    if probabilities.ndim == 4 and probabilities.shape[1] == 1:
        probabilities = probabilities[:, 0]
    if probabilities.ndim != 3 or probabilities.shape[1:] != (72, 72):
        raise HTTPException(500, f'Unexpected regional shape: {probabilities.shape}')
    if len(timestamps) != len(probabilities):
        raise HTTPException(500, 'Regional timestamps length mismatch')
    return probabilities, timestamps


def iso_time(value):
    return pd.Timestamp(value).tz_localize('UTC').isoformat().replace('+00:00', 'Z') if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert('UTC').isoformat().replace('+00:00', 'Z')


@app.get('/api/v1/status')
def status():
    return {
        'data_mode': 'historical_replay',
        'live_forecast_available': False,
        'global_baseline_ready': (BASE / 'test_predictions.csv').is_file(),
        'global_fusion_ready': (FUSION / 'test_predictions.csv').is_file(),
        'regional_ready': (REGION / 'test_probability_maps.npz').is_file(),
        'regional_probability_status': 'uncalibrated',
        'geographic_mapping_status': 'unverified',
    }


@app.get('/api/v1/forecast/global')
def global_forecast(model: str = Query('baseline', pattern='^(baseline|fusion)$'), index: int = Query(0, ge=0)):
    df = read_predictions(model)
    if index >= len(df):
        raise HTTPException(404, f'Index out of range (0..{len(df)-1})')
    row = df.iloc[index]
    t = pd.Timestamp(row['timestamp_utc'])
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    metrics = read_json((BASE if model == 'baseline' else FUSION) / 'metrics.json')
    return {
        'data_mode': 'historical_replay', 'sample_index': index, 'sample_count': len(df),
        'model': 'stormnet_v2_' + model, 'forecast_time_utc': iso_time(t),
        'target_time_utc': iso_time(t + pd.Timedelta(minutes=60)), 'horizon_minutes': 60,
        'current_sym_h_nT': float(row['persistence_sym_h_nT']),
        'predicted_sym_h_nT': float(row['predicted_sym_h_nT']),
        'actual_future_sym_h_nT': float(row['actual_sym_h_nT']),
        'metrics': {'test_mae_nT': metrics['test']['mae'], 'test_rmse_nT': metrics['test']['rmse']},
    }


@app.get('/api/v1/grid/metadata')
def grid_metadata():
    return {
        'rows': 72, 'columns': 72,
        'mapping_status': 'unverified', 'coordinates_available': False,
        'note': 'Do not place cells on the globe until IonoBench grid coordinates and orientation are verified.',
    }


@app.get('/api/v1/forecast/regional')
def regional_forecast(index: int = Query(0, ge=0)):
    maps, times = read_maps()
    if index >= len(maps):
        raise HTTPException(404, f'Index out of range (0..{len(maps)-1})')
    metrics = read_json(REGION / 'metrics.json')
    grid = maps[index].astype(np.float32)
    if not np.isfinite(grid).all():
        raise HTTPException(500, 'Nonfinite regional scores')
    return {
        'data_mode': 'historical_replay', 'sample_index': index, 'sample_count': len(maps),
        'model': 'stormnet_regional_2h', 'forecast_time_utc': iso_time(times[index]),
        'target_time_utc': iso_time(pd.Timestamp(times[index]) + pd.Timedelta(hours=2)),
        'horizon_minutes': 120,
        'target_definition': metrics['target'],
        'event_threshold': metrics['threshold'],
        'probability_status': 'uncalibrated',
        'display_label': 'TEC disturbance model score',
        'grid': {'rows': 72, 'columns': 72, 'mapping_status': 'unverified'},
        'scores': grid.tolist(),
        'warning': 'Not a geographic storm or power outage probability. Grid location is not yet verified.',
    }


@app.get('/api/v1/metrics')
def metrics():
    return {
        'global_baseline': read_json(BASE / 'metrics.json'),
        'global_fusion': read_json(FUSION / 'metrics.json'),
        'regional': read_json(REGION / 'metrics.json'),
    }

# --- Stateless historical replay API (v1.1) ---
# The client owns the cursor; GET requests never mutate server state.

def replay_sources():
    maps, timestamps = read_maps()
    if not len(maps):
        raise HTTPException(503, 'No regional replay samples available')
    return maps, timestamps


def replay_frame(index: int, model: str = 'baseline'):
    maps, timestamps = replay_sources()
    if not 0 <= index < len(maps):
        raise HTTPException(404, f'Index out of range (0..{len(maps)-1})')
    regional = regional_forecast(index=index)
    # The global and regional sample indices are not guaranteed to align.
    # Match by timestamp instead, and explicitly return null if no match exists.
    df = read_predictions(model)
    anchor = pd.Timestamp(timestamps[index])
    anchor = anchor.tz_localize('UTC') if anchor.tzinfo is None else anchor.tz_convert('UTC')
    matching = df.index[pd.to_datetime(df['timestamp_utc'], utc=True) == anchor]
    global_data = global_forecast(model=model, index=int(matching[0])) if len(matching) else None
    return {
        'data_mode': 'historical_replay',
        'playback_status': 'paused',
        'sample_index': index,
        'sample_count': len(maps),
        'forecast_time_utc': regional['forecast_time_utc'],
        'global_forecast': global_data,
        'regional_forecast': regional,
        'global_available_for_timestamp': global_data is not None,
        'geographic_mapping_status': 'unverified',
        'probability_status': 'uncalibrated',
    }


@app.get('/api/v1/replay/current')
def replay_current(index: int = Query(0, ge=0), model: str = Query('baseline', pattern='^(baseline|fusion)$')):
    return replay_frame(index, model)


@app.get('/api/v1/replay/next')
def replay_next(index: int = Query(0, ge=0), loop: bool = Query(True), model: str = Query('baseline', pattern='^(baseline|fusion)$')):
    maps, _ = replay_sources()
    if index >= len(maps):
        raise HTTPException(404, f'Index out of range (0..{len(maps)-1})')
    next_index = index + 1
    wrapped = next_index >= len(maps)
    if wrapped and loop:
        next_index = 0
    if wrapped and not loop:
        next_index = index
    frame = replay_frame(next_index, model)
    frame['previous_index'] = index
    frame['wrapped'] = wrapped and loop
    frame['end_of_replay'] = wrapped and not loop
    return frame


@app.get('/api/v1/replay/timeline')
def replay_timeline(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
    _, timestamps = replay_sources()
    count = len(timestamps)
    items = [{'index': i, 'forecast_time_utc': iso_time(timestamps[i])} for i in range(offset, min(offset + limit, count))]
    return {
        'data_mode': 'historical_replay',
        'sample_count': count,
        'offset': offset,
        'limit': limit,
        'has_more': offset + limit < count,
        'next_offset': offset + limit if offset + limit < count else None,
        'items': items,
    }
