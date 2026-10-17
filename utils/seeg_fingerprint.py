import functools
import inspect
import json
import random
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

#from net1d import Net1D

# Time blocks (seconds) of run 1, separated by 5 s gaps so no window crosses a boundary.
SPLITS = {'train': (0, 180), 'val': (185, 240), 'test': (245, 300)}
TRAIN_STRIDE_SEC = 0.1


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# DATA
def good_channels(tsv_path, edf_paths):
    """
    Contacts marked 'good' in the TSV and present in every EDF.
    """
    df = pd.read_csv(tsv_path, sep='\t').dropna(subset=['name'])
    is_good = df['status'].astype(str).str.strip().str.lower() == 'good'
    keep = list(dict.fromkeys(df.loc[is_good, 'name']))
    for p in edf_paths:
        present = mne.io.read_raw_edf(p, preload=False, verbose=False).ch_names
        keep = [c for c in keep if c in present]
    return keep


def load_run(path, channels, fs, avg_ref=True):
    """
    Return (n_channels, n_samples) float32 in microvolts.
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick(channels)
    raw.reorder_channels(channels)
    # remove DC and slow drift
    raw.filter(l_freq=1.0, h_freq=None, verbose=False)            
    nyq = raw.info['sfreq'] / 2
    notch = [f for f in (60, 120, 180, 240) if f < nyq - 2]
    if notch:
        raw.notch_filter(notch, verbose=False)
    if raw.info['sfreq'] != fs:
        raw.resample(fs, verbose=False)
    data = raw.get_data()
    if avg_ref:
        data = data - data.mean(axis=0, keepdims=True)
    return (data * 1e6).astype(np.float32)


def make_windows(data, start, stop, win, stride):
    """
    (C, T) to X (C*n, win), y (C*n,) with labels = channel index.
    """
    w = sliding_window_view(data[:, start:stop], win, axis=1)[:, ::stride]
    C, n, _ = w.shape
    return w.reshape(C * n, win), np.repeat(np.arange(C), n)


def build_loaders(run1_path, run2_path, channels, fs=256, win_sec=1.0,
                  norm='global', avg_ref=True, batch_size=128):
    """
    Returns (loaders, win). Loader keys: train, val, test (run 1) and run2.
 
    norm='global':  one mean/std from the training block (keeps amplitude differences).
    norm='channel': mean/std per contact from the training block. This uses each contact's own
                    statistics.
    norm='window':  z-score every window (removes amplitude).
    """
    win = int(round(win_sec * fs))
    stride = max(1, int(round(TRAIN_STRIDE_SEC * fs)))
    sec = lambda s: int(round(s * fs))
 
    run1 = load_run(run1_path, channels, fs, avg_ref)
    run2 = load_run(run2_path, channels, fs, avg_ref)
    if run1.shape[1] < sec(SPLITS['test'][1]):
        raise ValueError(f'run 1 is only {run1.shape[1] / fs:.0f}s long; splits need {SPLITS["test"][1]}s')
 
    # stats from training block only
    if norm in ('global', 'channel'):
        a, b = SPLITS['train']
        block = run1[:, sec(a):sec(b)]
        if norm == 'global':
            mu, sd = block.mean(), block.std()
        else:
            mu, sd = block.mean(1, keepdims=True), block.std(1, keepdims=True) + 1e-8
        run1, run2 = (run1 - mu) / sd, (run2 - mu) / sd
    elif norm != 'window':
        raise ValueError("norm must be 'global', 'channel' or 'window'")
 
    sets = {name: make_windows(run1, sec(a), sec(b), win, stride if name == 'train' else win)
            for name, (a, b) in SPLITS.items()}
    sets['run2'] = make_windows(run2, 0, run2.shape[1], win, win)
 
    loaders = {}
    for name, (X, y) in sets.items():
        if norm == 'window':
            X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)   # (N, 1, win)
        loaders[name] = DataLoader(TensorDataset(X, torch.from_numpy(y)),
                                   batch_size=batch_size, shuffle=(name == 'train'))
    return loaders, win



# MODELS
def _tracked(fn):
    """
    Track which builder and arguments made a model, for save_model
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        model = fn(*args, **kwargs)
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        model.build_info = {'builder': fn.__name__, 'kwargs': dict(bound.arguments)}
        return model
    return wrapper

@_tracked
def build_mlp(win, n_classes, hidden=(512, 256)):
    """Two hidden layers, ReLU, no dropout."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(win, hidden[0]), nn.ReLU(),
        nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        nn.Linear(hidden[1], n_classes),
    )

@_tracked
def build_net1d(n_classes):
    """
    Starter config for 256 Hz, 1-2 s windows. Length: 256 - 128 - 64 - 32 - 16.
    Each filter count must be divisible by groups_width and by 2.
    """
    return Net1D(in_channels=1, base_filters=32, ratio=1.0,
                 filter_list=[32, 64, 128], m_blocks_list=[1, 2, 2],
                 kernel_size=16, stride=2, groups_width=8, n_classes=n_classes)

class Lambda(nn.Module):
    """
    Helper function
    """
    def __init__(self, fn, desc=''):
        super().__init__()
        self.fn = fn
        self.desc = desc

    def forward(self, x):
        return self.fn(x)

    def extra_repr(self):
        return self.desc

@_tracked
def build_simple_cnn(win, n_classes, in_channels=1, out_channels=64, n_len_seg=64, kernel_size=16, stride=2, n_convs=2, dropout=0.1):
    """
    Simple 1D CNN
    """
    assert win % n_len_seg == 0, 'win must be divisible by n_len_seg'
    assert out_channels % 8 == 0, 'out_channels must be divisible by nhead=8'
    n_seg = win // n_len_seg

    convs, in_ch, length = [], in_channels, n_len_seg
    for i in range(n_convs):
        assert length >= kernel_size, (f'conv {i + 1} would get only {length} samples (kernel_size={kernel_size}): '
                                       'use fewer convs, a smaller kernel/stride, or a longer n_len_seg')
        convs += [nn.Conv1d(in_ch, out_channels, kernel_size, stride), nn.ReLU()]
        in_ch, length = out_channels, (length - kernel_size) // stride + 1
    encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=8, dim_feedforward=128, dropout=dropout, batch_first=True)

    return nn.Sequential(
        Lambda(lambda x: x.reshape(-1, in_channels, n_seg, n_len_seg)
              .permute(0, 2, 1, 3).reshape(-1, in_channels, n_len_seg), f'split into {n_seg} segments: (N, {in_channels}, {win}) -> (N*{n_seg}, {in_channels}, {n_len_seg})'),
        *convs,
        Lambda(lambda x: x.mean(-1), 'mean over time'),
        Lambda(lambda x: x.reshape(-1, n_seg, out_channels), f'group segments by window: (N*{n_seg}, {out_channels}) -> (N, {n_seg}, {out_channels})'),
        nn.TransformerEncoder(encoder_layer, num_layers=1),
        Lambda(lambda x: x.mean(1), 'mean over segments'),
        nn.Linear(out_channels, n_classes),
    )

    
# TRAIN & EVAL
@torch.no_grad()
def evaluate(model, loader):
    """Returns (accuracy, mean cross-entropy loss)."""
    device = next(model.parameters()).device
    model.eval()
    correct = loss = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction='sum').item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return correct / n, loss / n


def train_model(model, loaders, epochs=60, lr=1e-3, weight_decay=1e-2, patience=10):
    """
    Trains with AdamW; early stopping and model selection use the val loader only.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best_acc, best_state, bad = [], -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        total = n = 0
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)

        val_acc, val_loss = evaluate(model, loaders['val'])
        history.append({'epoch': epoch, 'train_loss': total / n, 'val_loss': val_loss, 'val_acc': val_acc})
        print(f'epoch {epoch:3d}  train_loss {total / n:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f'Early stopping at epoch {epoch}, best val_acc: {best_acc:.3f}')
                break

    model.load_state_dict(best_state)
    return model, history


def save_results(path, name, model, history, loaders, n_classes, config=None):
    """
    Evaluate the final model on all four splits and merge the results into a JSON file.
    """
    entry = {
        'history': history,
        'best_epoch': max(history, key=lambda h: h['val_acc'])['epoch'],
        'metrics': {f'{s}_acc': evaluate(model, loaders[s])[0] for s in ('train', 'val', 'test', 'run2')},
        'n_params': sum(p.numel() for p in model.parameters()),
        'n_classes': n_classes,
        'config': config or {},
    }
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = entry
    path.write_text(json.dumps(data, indent=2))


BUILDERS = {f.__name__: f for f in (build_mlp, build_net1d, build_simple_cnn)}

def save_model(path, model):
    """
    Save weights and builder name/arguments
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({**model.build_info, 'state_dict': state}, path)

def load_model(path):
    """
    Rebuild a model
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = BUILDERS[ckpt['builder']](**ckpt['kwargs'])
    model.load_state_dict(ckpt['state_dict'])
    return model.eval()

def export_patient(patient_dir, channels, loaders):
    """
    Save contact names and eact test/run2 windows the model sees
    """
    d = Path(patient_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'channels.json').write_text(json.dumps(list(channels)))
    arrays = {}
    for split in ('test', 'run2'):
        X, y = loaders[split].dataset.tensors
        arrays[f'{split}_X'] = X.squeeze(1).numpy().astype(np.float32)
        arrays[f'{split}_y'] = y.numpy()
    np.savez_compressed(d / 'windows.npz', **arrays)