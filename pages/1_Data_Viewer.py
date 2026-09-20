from pathlib import Path
import numpy as np
import pandas as pd
import mne
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "example_data"

def list_subjects(data_dir):
    return sorted(p.name for p in data_dir.glob("sub-*") if p.is_dir())

def find_recordings(subject_dir):
    return sorted(subject_dir.rglob("*_ieeg.edf"))

def channels_path_for(edf_path):
    return edf_path.with_name(edf_path.name.replace("_ieeg.edf", "_channels.tsv"))


@st.cache_resource
def load_raw(edf_path, good_channels, target_sample_rate=128, max_duration_seconds=30):
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    # Only pick channels with good status
    raw.pick(good_channels)
    # Crop length to 30 seconds
    raw.crop(tmax=min(max_duration_seconds, raw.times[-1]))
    # Resample to lower resolution
    raw.resample(target_sample_rate)
    # Perform average referencing
    raw.set_eeg_reference(ref_channels="average")
    return raw

@st.cache_resource
def load_metadata(tsv_path):
    # Only select good channels using metadata
    good_channels = []
    df = pd.read_csv(tsv_path, sep='\t')
    for index, row in df.iterrows():
        if (row['status'].strip().lower() == 'good') and (row['name'] not in good_channels):
            good_channels.append(row['name'])
    return good_channels
    
    
st.title("Data Viewer")
st.markdown(
    """
    <div style="text-align: left;">
        Select channels you want to view and the time window from the dropdown and slider on the left.
        Raw SEEG signals are in microvolts. They have been downsampled from 500 Hz to 128 Hz, clipped to 
        30 seconds, and average referencing has been applied.
    </div>
    """,
    unsafe_allow_html=True
)

# Subject selection
subjects = list_subjects(DATA_DIR)
if not subjects:
    st.error(f"No subject folders found in {DATA_DIR}")
    st.stop()

subject = st.sidebar.selectbox("Subject", subjects)

recordings = find_recordings(DATA_DIR / subject)
if not recordings:
    st.warning(f"No EDF files found for {subject}")
    st.stop()
    
edf_path = st.sidebar.selectbox("Recording", recordings, format_func=lambda p: p.name)
tsv_path = channels_path_for(edf_path)

good_channels = load_metadata(tsv_path)
raw = load_raw(edf_path, good_channels)

selected = st.sidebar.multiselect(
    "Channels", raw.ch_names, default=raw.ch_names[:3]
)

if not selected:
    st.info("Select at least one channel to plot")
    st.stop()
    
duration = float(raw.times[-1])
t_range = st.sidebar.slider(
    "Time window (s)",
    min_value=0.0,
    max_value = duration,
    value=(0.0, float(min(30, duration))),
    step=0.5
)

start, stop = raw.time_as_index(list(t_range))
data, times = raw.get_data(
    picks=selected, start=start, stop=stop, return_times=True, units="uV"
)

n = len(selected)
fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.2 / n)

for i, name in enumerate(selected, start=1):
    fig.add_trace(
        go.Scattergl(x=times, y=data[i-1], mode="lines", name=name),
        row=i, col=1
    )
    fig.update_yaxes(title_text=name, row=i, col=1)

fig.update_xaxes(title_text="Time (s)", row=n, col=1)
fig.update_layout(height=150 * n + 80, showlegend=False,
                  margin=dict(l=60, r=20, t=20, b=40))

st.plotly_chart(fig, width="stretch")