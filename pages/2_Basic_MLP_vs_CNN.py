import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
# so `import seeg_fingerprint` works from pages/
#sys.path.insert(0, str(ROOT))                       
from utils.seeg_fingerprint import load_model             

RESULTS_DIR = ROOT / 'results'
MODEL_ORDER = ['MLP', 'Net1D', 'CNN']

#st.set_page_config(page_title='SEEG fingerprinting', layout='wide')
st.title('Basic MLP versus Simple CNN for Fingerprinting')


@st.cache_data
def load_windows(path, mtime):
    z = np.load(path)
    return {k: z[k] for k in z.files}


@st.cache_resource
def get_model(path, mtime):                         
    return load_model(path)


@torch.no_grad()
def predict(model, window):
    x = torch.from_numpy(window).float().reshape(1, 1, -1)
    return torch.softmax(model(x), dim=1)[0].numpy()


# patient
patients = sorted(p.name for p in RESULTS_DIR.glob('*') if (p / 'results.json').exists())
if not patients:
    st.error(f'No results found in {RESULTS_DIR}. Expected results/<patient>/results.json.')
    st.stop()

patient = st.sidebar.selectbox('Patient', patients)
pdir = RESULTS_DIR / patient
results = json.loads((pdir / 'results.json').read_text())
names = [m for m in MODEL_ORDER if m in results] + [m for m in results if m not in MODEL_ORDER]
st.subheader(patient)

# loss curves
st.header('Training and validation loss')
fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4), sharey=True, squeeze=False)
for ax, name in zip(axes[0], names):
    hist = pd.DataFrame(results[name]['history'])
    ax.plot(hist['epoch'], hist['train_loss'], label='Train')
    ax.plot(hist['epoch'], hist['val_loss'], label='Validation')
    ax.axvline(results[name]['best_epoch'], color='gray', linestyle='--', linewidth=1,
               label='Best val. accuracy')
    ax.set_title(name)
    ax.set_xlabel('Epoch')
    ax.grid(alpha=0.3)
axes[0][0].set_ylabel('Cross-entropy loss')
axes[0][0].legend()
fig.tight_layout()
st.pyplot(fig)
#st.caption('Train loss is the running average over each epoch in training mode '
#           '(dropout active where used); validation loss is computed in eval mode. '
#           'The reported model is the one at the dashed line.')

# accuracy table
st.header('Accuracy')
rows = []
for name in names:
    m = results[name]['metrics']
    rows.append({'Model': name, 'Train': m['train_acc'], 'Test': m['test_acc'],
                 'Run 2': m['run2_acc'], 'Parameters': results[name]['n_params']})
table = pd.DataFrame(rows).set_index('Model')
st.dataframe(
    table.style.format({'Train': '{:.1%}', 'Test': '{:.1%}', 'Run 2': '{:.1%}', 'Parameters': '{:,}'}),
    width='stretch',
)
n_classes = results[names[0]]['n_classes']
st.caption(f'Chance level: {1 / n_classes:.1%} ({n_classes} contacts). '
           'Test = held-out time block of run 1; Run 2 = a separate recording.')
with st.expander('Settings used'):
    st.json({name: results[name]['config'] for name in names})

# Model architecture
models = {n: pdir / 'models' / f'{n}.pt' for n in names if (pdir / 'models' / f'{n}.pt').exists()}
st.header('Model architecture')
if models:
    for tab, (name, path) in zip(st.tabs(list(models)), models.items()):
        model = get_model(str(path), path.stat().st_mtime)
        info = model.build_info
        with tab:
            st.code(str(model), language=None)
            args = ', '.join(f'{k}={v}' for k, v in info['kwargs'].items())
            st.caption(f"{info['builder']}({args})")
else:
    st.info('No saved models for this patient yet.')


# Let the user try a prediction
st.header('Try a prediction')
models = {n: pdir / 'models' / f'{n}.pt' for n in names if (pdir / 'models' / f'{n}.pt').exists()}
ch_path, win_path = pdir / 'channels.json', pdir / 'windows.npz'

if not (models and ch_path.exists() and win_path.exists()):
    st.info('No saved models or windows for this patient yet. In the notebook, call '
            'sf.export_patient(...) and sf.save_model(...) for each model.')
else:
    channels = json.loads(ch_path.read_text())
    windows = load_windows(str(win_path), win_path.stat().st_mtime)
    fs = next((results[n]['config'].get('fs') for n in names if results[n]['config'].get('fs')), None)

    c1, c2, c3 = st.columns([2, 2, 1])
    contact = c1.selectbox('Electrode contact', channels)
    source = c2.radio('Window source', ['Run 2', 'Run 1 test block'], horizontal=True)
    redraw = c3.button('Draw another window')

    split = 'run2' if source == 'Run 2' else 'test'
    X, y = windows[f'{split}_X'], windows[f'{split}_y']
    ci = channels.index(contact)
    candidates = np.flatnonzero(y == ci)

    if len(candidates) == 0:
        st.warning(f'{source} has no windows for {contact}.')
    else:
        # Keep the same window across Streamlit reruns; draw a new one when the choice changes or on click.
        key = (patient, split, ci)
        if redraw or st.session_state.get('draw_key') != key:
            st.session_state['draw_key'] = key
            st.session_state['draw_idx'] = int(np.random.choice(candidates))
        idx = st.session_state['draw_idx']
        window = X[idx]

        fig, ax = plt.subplots(figsize=(9, 2.5))
        ax.plot(np.arange(len(window)) / fs if fs else np.arange(len(window)), window, linewidth=0.8)
        ax.set_xlabel('Time (s)' if fs else 'Sample')
        ax.set_ylabel('Normalized amplitude')
        ax.set_title(f'{contact} ({source}), window {idx}')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

        cols = st.columns(len(models))
        for col, (name, path) in zip(cols, models.items()):
            probs = predict(get_model(str(path), path.stat().st_mtime), window)
            order = np.argsort(probs)[::-1]
            pred = int(order[0])
            rank = int(np.where(order == ci)[0][0]) + 1
            with col:
                st.subheader(name)
                st.metric('Predicted contact', channels[pred])
                if pred == ci:
                    st.success(f'Correct ({probs[pred]:.1%} confidence)')
                else:
                    st.error(f'Wrong: true contact {contact} ranked {rank} of {len(channels)}')
                top = pd.DataFrame({'Contact': [channels[i] for i in order[:5]],
                                    'Probability': probs[order[:5]]})
                st.dataframe(top.style.format({'Probability': '{:.1%}'}), hide_index=True, width='stretch')