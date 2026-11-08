
from pathlib import Path
import pickle
import numpy as np

DATA_PATH = Path(
    "data/raw/ionobench/"
    "IonoBench_chronologicalSplit.pickle"
)

def inspect(value, name="root", depth=0):
    indent = "  " * depth

    print(f"{indent}{name}: {type(value).__name__}")

    if isinstance(value, dict):
        print(f"{indent}Keys: {list(value.keys())[:30]}")
        if depth < 2:
            for key, item in list(value.items())[:15]:
                inspect(item, str(key), depth + 1)

    elif isinstance(value, np.ndarray):
        print(f"{indent}Shape: {value.shape}")
        print(f"{indent}Dtype: {value.dtype}")

    elif isinstance(value, (list, tuple)):
        print(f"{indent}Length: {len(value)}")
        if len(value) > 0 and depth < 2:
            inspect(value[0], "First item", depth + 1)

    else:
        shape = getattr(value, "shape", None)
        if shape is not None:
            print(f"{indent}Shape: {shape}")

def main():
    print("Loading IonoBench...")
    print("File:", DATA_PATH.resolve())

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    # Only load pickle files from trusted sources.
    with DATA_PATH.open("rb") as file:
        data = pickle.load(file)

    print("\n=== Dataset Structure ===")
    inspect(data)

    print("\nInspection completed.")

if __name__ == "__main__":
    main()
