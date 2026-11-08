
from pathlib import Path
from collections import Counter
import pickle
import numpy as np
import pandas as pd

DATA_PATH = Path(
    "data/raw/ionobench/"
    "IonoBench_chronologicalSplit.pickle"
)

print("Loading IonoBench...")

# Load only files from trusted sources.
with DATA_PATH.open("rb") as file:
    data = pickle.load(file)

dates = pd.to_datetime(data["Dates"], utc=True)

print("\n=== TIME RANGE ===")
print("Start:", dates.min())
print("End:", dates.max())
print("Sorted:", dates.is_monotonic_increasing)
print("Duplicate timestamps:", dates.duplicated().sum())

print("\n=== TIME RESOLUTION ===")
intervals = pd.Series(dates).diff().dropna()
print(intervals.value_counts().head(10))

print("\n=== OMNI FEATURE NAMES ===")
print(data["OMNI_Names"].to_string(index=False))

print("\n=== OMNI DATA SAMPLE ===")
omni = data["NormOMNI"]
print("Shape:", omni.shape)
print("Dtype:", omni.dtype)

for i in range(min(5, len(omni))):
    print(f"Row {i}:", omni[i, :10])

print("\n=== OMNI VALUE TYPES ===")
types = Counter(
    type(value).__name__
    for value in omni[:100].flat
)
print(types)

print("\n=== DATA SPLITS ===")
for name, values in data["split_dates"].items():
    print(f"\n{name}:")
    print("Count:", len(values))
    print("First values:", values[:3])
    print("Last values:", values[-3:])

print("\n=== TEC STATISTICS ===")
tec = data["NormTEC"]
print("Shape:", tec.shape)
print("Min:", np.nanmin(tec))
print("Max:", np.nanmax(tec))
print("NaN count:", np.isnan(tec).sum())

print("\nInspection completed.")
