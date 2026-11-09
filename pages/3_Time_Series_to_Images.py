import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / 'results_hw3'
FIG_DIR = ROOT / 'report' / 'figures'

# folder under results_hw3/
WINDOW_DIRS = {'1-second windows': '1_sec_results', '2-second windows': '2_sec_results'}

# Keys as saved in results.json
MODEL_ORDER = ['TwoBranch', 'TSOnly', 'GridOnly']
LABELS = {'TwoBranch': 'Two-branch', 'TSOnly': 'GAF/MTF only', 'GridOnly': 'Lag grids only'}

# Model diagrams
FIG_KEYWORDS = {'TwoBranch': ['twobranch'], 'TSOnly': ['tsonly'], 'GridOnly': ['gridonly']}
FIG_OVERRIDES = {}
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

#st.set_page_config(page_title='SEEG fingerprinting', layout='wide')
st.title('Time Series to Images for Fingerprinting')


@st.cache_data
def load_json(path, mtime):
    return json.loads(Path(path).read_text())


def ordered(names):
    names = list(names)
    return [m for m in MODEL_ORDER if m in names] + [m for m in names if m not in MODEL_ORDER]


def label(name):
    return LABELS.get(name, name)


def _norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def find_figure(name):
    if name in FIG_OVERRIDES:
        p = FIG_DIR / FIG_OVERRIDES[name]
        return p if p.exists() else None
    if not FIG_DIR.is_dir():
        return None
    keys = FIG_KEYWORDS.get(name, [_norm(name)])
    for f in sorted(FIG_DIR.iterdir()):
        if f.suffix.lower() in IMG_EXTS and any(k in _norm(f.stem) for k in keys):
            return f
    return None


def loss_figure(results):
    names = ordered(results)
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        hist = pd.DataFrame(results[name]['history'])
        ax.plot(hist['epoch'], hist['train_loss'], label='Train')
        ax.plot(hist['epoch'], hist['val_loss'], label='Validation')
        ax.axvline(results[name]['best_epoch'], color='gray', linestyle='--', linewidth=1,
                   label='Best val. accuracy')
        ax.set_title(label(name))
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel('Cross-entropy loss')
    axes[0][0].legend()
    fig.tight_layout()
    return fig


# patient
patients = sorted({p.name
                   for d in WINDOW_DIRS.values() if (RESULTS_ROOT / d).is_dir()
                   for p in (RESULTS_ROOT / d).glob('*') if (p / 'results.json').exists()})
if not patients:
    st.error(f'No results found in {RESULTS_ROOT}. Expected '
             f'results_hw3/<1_sec_results|2_sec_results>/<patient>/results.json.')
    st.stop()

patient = st.sidebar.selectbox('Patient', patients)
st.subheader(patient)

all_results = {}
for title, d in WINDOW_DIRS.items():
    p = RESULTS_ROOT / d / patient / 'results.json'
    if p.exists():
        all_results[title] = load_json(str(p), p.stat().st_mtime)

# loss curves
st.header('Training and validation loss')
for title, d in WINDOW_DIRS.items():
    st.subheader(title)
    if title not in all_results:
        st.info(f'No results for {patient} in {d}.')
        continue
    fig = loss_figure(all_results[title])
    st.pyplot(fig)
    plt.close(fig)

# accuracy tables
st.header('Accuracy')
cols = st.columns(len(WINDOW_DIRS))
for col, title in zip(cols, WINDOW_DIRS):
    with col:
        st.subheader(title)
        results = all_results.get(title)
        if results is None:
            st.info('No results.')
            continue
        rows = []
        for name in ordered(results):
            m = results[name]['metrics']
            rows.append({'Model': label(name), 'Train': m['train_acc'], 'Test': m['test_acc'],
                         'Run 2': m['run2_acc'], 'Parameters': results[name]['n_params']})
        table = pd.DataFrame(rows).set_index('Model')
        st.dataframe(
            table.style.format({'Train': '{:.1%}', 'Test': '{:.1%}', 'Run 2': '{:.1%}',
                                'Parameters': '{:,}'}),
            width='stretch',
        )

if all_results:
    first = next(iter(all_results.values()))
    n_classes = first[ordered(first)[0]]['n_classes']
    st.caption(f'Chance level: {1 / n_classes:.1%} ({n_classes} contacts). '
               'Test = held-out time block of run 1; Run 2 = a separate recording.')
    with st.expander('Settings used'):
        for title, results in all_results.items():
            st.markdown(f'**{title}**')
            st.json({name: results[name]['config'] for name in ordered(results)})

# model architecture
st.header('Model architecture')
model_names = ordered({n for r in all_results.values() for n in r})
if model_names:
    for tab, name in zip(st.tabs([label(n) for n in model_names]), model_names):
        with tab:
            fig_path = find_figure(name)
            if fig_path:
                st.image(str(fig_path))
            else:
                found = sorted(f.name for f in FIG_DIR.iterdir()) if FIG_DIR.is_dir() else []
                st.info(f'No diagram found for {name} in {FIG_DIR}. '
                        f'Files there: {found if found else "none"}. '
                        'Rename the file to contain the model name or set FIG_OVERRIDES at the top of this page.')