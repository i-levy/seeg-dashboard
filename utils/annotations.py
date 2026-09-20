from datetime import datetime
from pathlib import Path
import pandas as pd

ANNOTATIONS_DIR = Path(__file__).resolve().parents[1] / "annotations"
COLUMNS = ["onset", "duration", "channel", "label", "note", "created_at"]

def annotations_path_for(edf_path):
    # Creates an annotation CSV for each recording
    name = edf_path.name.replace("_ieeg.edf", "_annotations.csv")
    return ANNOTATIONS_DIR / name

def load_annotations(edf_path):
    path = annotations_path_for(edf_path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path).fillna("")
    if "channel" not in df.columns:
        df["channel"] = ""
    return df

def add_annotation(edf_path, onset, duration, channel, label, note):
    path = annotations_path_for(edf_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "onset": onset,
        "duration": duration,
        "channel": channel,
        "label": label,
        "note": note,
        "created_at": datetime.now().isoformat(timespec="seconds")
    }])
    # Append one row
    row.to_csv(path, mode="a", header=not path.exists(), index=False)

def delete_annotation(edf_path, index):
    df = load_annotations(edf_path).drop(index=index)
    df.to_csv(annotations_path_for(edf_path), index=False)