import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent.parent          # dashboard/ (this file lives in pages/)
RESULTS_ROOT = ROOT / 'results_hw3'
FIG_DIR = ROOT / 'report' / 'figures'

# display title -> folder under results_hw3/
#WINDOW_DIRS = {'1-second windows': '1_sec_results', '2-second windows': '2_sec_results'}
WINDOW_DIRS = {'1-second windows': '1_sec_results'}
# window length used for the cross-subject summary plot
SUMMARY_DIR = '1_sec_results'

# keys as saved in results.json
MODEL_ORDER = ['TwoBranch', 'TSOnly', 'GridOnly']
LABELS = {'TwoBranch': 'Two-branch', 'TSOnly': 'GAF/MTF only', 'GridOnly': 'Lag grids only'}

# How to find each model's diagram in report/figures: a file whose name contains one of these
# (case and punctuation ignored). To point at a specific file instead, fill FIG_OVERRIDES,
# e.g. {'TwoBranch': 'BranchNet.png'}.
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


@st.cache_data
def n_windows(path, mtime):
    """Number of test / run-2 windows (labels only, so the large arrays are not read)."""
    z = np.load(path)
    return {'test': int(z['test_y'].shape[0]), 'run2': int(z['run2_y'].shape[0])}


def wilson(p, n, z=1.96):
    """95% Wilson score interval for a proportion p measured on n samples."""
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def summary_frame():
    """One row per (subject, model): test / run-2 accuracy and the number of windows behind each."""
    rows = []
    for pdir in sorted((RESULTS_ROOT / SUMMARY_DIR).glob('*')):
        rf = pdir / 'results.json'
        if not rf.exists():
            continue
        res = load_json(str(rf), rf.stat().st_mtime)
        wf = pdir / 'windows.npz'
        n = n_windows(str(wf), wf.stat().st_mtime) if wf.exists() else {}
        for name in ordered(res):
            m = res[name]['metrics']
            rows.append({'Subject': pdir.name, 'Model': label(name), 'n_classes': res[name]['n_classes'],
                         'Test': m['test_acc'], 'Run2': m['run2_acc'],
                         'n_test': n.get('test', np.nan), 'n_run2': n.get('run2', np.nan)})
    return pd.DataFrame(rows)


def summary_figure(df, show_ci=True):
    subjects = list(df['Subject'].unique())
    present = set(df['Model'])
    order = [LABELS[m] for m in MODEL_ORDER if LABELS[m] in present] + \
            [m for m in df['Model'].unique() if m not in LABELS.values()]
    chance = 1 / df.groupby('Subject')['n_classes'].first()
    x = np.arange(len(subjects))
    width = 0.8 / len(order)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = [('Test', 'n_test', 'Test (held-out block of run 1)'),
              ('Run2', 'n_run2', 'Run 2 (separate recording)')]
    for ax, (metric, nkey, title) in zip(axes, panels):
        for i, model in enumerate(order):
            sub = df[df['Model'] == model].set_index('Subject').reindex(subjects)
            p = sub[metric].to_numpy(float)
            yerr = None
            if show_ci:
                n = sub[nkey].to_numpy(float)
                ok = np.isfinite(n) & np.isfinite(p) & (n > 0)
                below, above = np.zeros_like(p), np.zeros_like(p)
                lo, hi = wilson(p[ok], n[ok])
                below[ok], above[ok] = p[ok] - lo, hi - p[ok]
                yerr = np.clip(np.vstack([below, above]), 0, None)
            ax.bar(x + (i - (len(order) - 1) / 2) * width, p, width, yerr=yerr, capsize=3,
                   error_kw={'linewidth': 1}, label=model)
        for j, s_ in enumerate(subjects):
            ax.hlines(chance[s_], j - 0.4, j + 0.4, colors='k', linestyles='--', linewidth=1,
                      label='Chance' if j == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Accuracy')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# patient
patients = sorted({p.name
                   for d in WINDOW_DIRS.values() if (RESULTS_ROOT / d).is_dir()
                   for p in (RESULTS_ROOT / d).glob('*') if (p / 'results.json').exists()})
if not patients:
    st.error(f'No results found in {RESULTS_ROOT}. Expected '
             f'results_hw3/<1_sec_results|2_sec_results>/<patient>/results.json.')
    st.stop()

# summary across subjects
st.header('Accuracy across subjects')
summary = summary_frame()
if summary.empty:
    st.info(f'No results found in {RESULTS_ROOT / SUMMARY_DIR}.')
else:
    show_ci = st.checkbox('Show 95% confidence intervals', value=True)
    missing = sorted(summary.loc[summary['n_test'].isna(), 'Subject'].unique())
    if show_ci and missing:
        st.warning(f'No windows.npz for {", ".join(missing)}, so no error bars are drawn for them.')
    fig = summary_figure(summary, show_ci)
    st.pyplot(fig)
    plt.close(fig)
    st.caption('1-second windows. Dashed lines mark chance level for each subject. Error bars are 95% Wilson '
               'intervals for each accuracy given the number of evaluation windows. They reflect sampling '
               'error of the evaluation set only, not variation between training runs (one run per model). '
               'Windows from the same contact are correlated, so the true uncertainty is larger.')
st.divider()

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