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
from pyts.image import GramianAngularField, MarkovTransitionField

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


def lag_grids(W, lag_samples, n_bins, lo, hi):
    """
    W: (B, win) -> (B, n_lags, n_bins, n_bins) pair counts; [b, k, lagged_bin, current_bin].
    """
    B = W.shape[0]
    W = np.clip(W, lo, hi)                                   # clip outliers into the edge bins
    edges = np.linspace(lo, hi, n_bins + 1)
    b = np.clip(np.searchsorted(edges, W, side="left") - 1, 0, n_bins - 1)
    out = np.zeros((B, len(lag_samples), n_bins, n_bins), np.float32)
    row = np.arange(B)[:, None] * n_bins**2
    for k, lag in enumerate(lag_samples):
        flat = (row + b[:, :-lag] * n_bins + b[:, lag:]).ravel()
        out[:, k] = np.bincount(flat, minlength=B * n_bins**2).reshape(B, n_bins, n_bins)
    return out

    
def make_windows(data, start, stop, win, stride):
    """
    (C, T) to X (C*n, win), y (C*n,) with labels = channel index.
    """
    w = sliding_window_view(data[:, start:stop], win, axis=1)[:, ::stride]
    C, n, _ = w.shape
    return w.reshape(C * n, win), np.repeat(np.arange(C), n)


def build_loaders(run1_path, run2_path, channels, splits=SPLITS, fs=256, win_sec=1.0,
                  train_stride_sec=TRAIN_STRIDE_SEC, norm='global', avg_ref=True, batch_size=128,
                  encoder=None, num_workers=4):
    """
    Returns (loaders, win). Loader keys: train, val, test (run 1) and run2.
 
    norm='global':  one mean/std from the training block (keeps amplitude differences).
    norm='channel': mean/std per contact from the training block. This uses each contact's own
                    statistics.
    norm='window':  z-score every window (removes amplitude).
    """
    win = int(round(win_sec * fs))
    stride = max(1, int(round(train_stride_sec * fs)))
    sec = lambda s: int(round(s * fs))
    encoder = encoder or ImageEncoder()

    run1 = load_run(run1_path, channels, fs, avg_ref)
    run2 = load_run(run2_path, channels, fs, avg_ref)
    if run1.shape[1] < sec(splits['test'][1]):
        raise ValueError(f'run 1 is only {run1.shape[1] / fs:.0f}s long; splits need {splits["test"][1]}s')

    if norm in ('global', 'channel'):
        a, b = splits['train']
        block = run1[:, sec(a):sec(b)]
        if norm == 'global':
            mu, sd = block.mean(), block.std()
        else:
            mu, sd = block.mean(1, keepdims=True), block.std(1, keepdims=True) + 1e-8
        run1, run2 = (run1 - mu) / sd, (run2 - mu) / sd
    elif norm != 'window':
        raise ValueError("norm must be 'global', 'channel' or 'window'")

    sets = {name: make_windows(run1, sec(a), sec(b), win, stride if name == 'train' else win)
            for name, (a, b) in splits.items()}
    sets['run2'] = make_windows(run2, 0, run2.shape[1], win, win)

    loaders = {}
    for name, (X, y) in sets.items():
        if norm == 'window':
            X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)),   # (N, win): no channel dim
                           torch.from_numpy(y))
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == 'train'),
                                   collate_fn=encoder.collate, num_workers=num_workers,
                                   persistent_workers=num_workers > 0)
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


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class Branch(nn.Module):
    """Small CNN -> fixed-length feature vector, independent of image size."""
    def __init__(self, in_ch, widths=(16, 32, 64)):
        super().__init__()
        chans = (in_ch,) + tuple(widths)
        self.convs = nn.Sequential(*[conv_block(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = widths[-1]

    def forward(self, x):
        return self.pool(self.convs(x)).flatten(1)

class TwoBranchNet(nn.Module):
    def __init__(self, n_classes, ts_channels=2, n_lags=3, widths=(16, 32, 64), dropout=0.1):
        super().__init__()
        self.ts_branch = Branch(ts_channels, widths)     # GAF + MTF (time x time)
        self.grid_branch = Branch(n_lags, widths)        # lag grids (amplitude x amplitude)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.ts_branch.out_dim + self.grid_branch.out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, ts_img, grid_img):
        z = torch.cat([self.ts_branch(ts_img), self.grid_branch(grid_img)], dim=1)
        return self.head(z)

    
class ImageEncoder:
    def __init__(self, size=64, mtf_bins=8, lag_samples=(2, 8, 32),
                 grid_bins=16, grid_range=(-4.0, 4.0), gaf_scale=None,
                 use_ts=True, use_grid=True):
        assert use_ts or use_grid, 'at least one of use_ts / use_grid must be True'
        self.size, self.mtf_bins = size, mtf_bins
        self.lag_samples, self.grid_bins = tuple(lag_samples), grid_bins
        self.grid_range, self.gaf_scale = grid_range, gaf_scale
        self.use_ts, self.use_grid = use_ts, use_grid
        self.gaf = GramianAngularField(image_size=size, method="summation",
                                       sample_range=None if gaf_scale else (-1, 1))
        self.mtf = MarkovTransitionField(image_size=size, n_bins=mtf_bins)

    def encode(self, X):                                     # X: (B, win) numpy
        X = np.asarray(X, np.float64)
        ts = grid = None
        if self.use_ts:
            Xg = np.clip(X / self.gaf_scale, -1, 1) if self.gaf_scale else X
            ts = np.stack([self.gaf.fit_transform(Xg), self.mtf.fit_transform(X)],
                          axis=1).astype(np.float32)
        if self.use_grid:
            grid = lag_grids(X, self.lag_samples, self.grid_bins, *self.grid_range)
            grid = (np.log1p(grid) / np.log1p(X.shape[1])).astype(np.float32)
        return ts, grid

    def collate(self, batch):                                # used as DataLoader collate_fn
        X = torch.stack([b[0] for b in batch]).numpy()
        y = torch.stack([b[1] for b in batch])
        arrays = [a for a in self.encode(X) if a is not None]
        return (*[torch.from_numpy(a) for a in arrays], y)   # 1 or 2 image tensors, then labels

    def config(self):
        return dict(size=self.size, mtf_bins=self.mtf_bins, lag_samples=list(self.lag_samples),
                    grid_bins=self.grid_bins, grid_range=list(self.grid_range),
                    gaf_scale=self.gaf_scale, use_ts=self.use_ts, use_grid=self.use_grid)

class SingleBranchNet(nn.Module):
    def __init__(self, n_classes, in_ch, widths=(16, 32, 64), dropout=0.1):
        super().__init__()
        self.branch = Branch(in_ch, widths)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.branch.out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.branch(x))

@_tracked
def build_ts_only(n_classes, ts_channels=2, widths=(16, 32, 64), dropout=0.1):
    return SingleBranchNet(n_classes, ts_channels, widths, dropout)

@_tracked
def build_grid_only(n_classes, n_lags=3, widths=(16, 32, 64), dropout=0.1):
    return SingleBranchNet(n_classes, n_lags, widths, dropout)


@_tracked
def build_two_branch(n_classes, ts_channels=2, n_lags=3, widths=(16, 32, 64), dropout=0.1):
    return TwoBranchNet(n_classes, ts_channels, n_lags, widths, dropout)
    
# TRAIN & EVAL
def _unpack(batch, device):
    *inputs, y = batch
    return [t.to(device) for t in inputs], y.to(device)

    
@torch.no_grad()
def evaluate(model, loader):
    """Returns (accuracy, mean cross-entropy loss)."""
    device = next(model.parameters()).device
    model.eval()
    correct = loss = n = 0
    for batch in loader:
        inputs, y = _unpack(batch, device)
        out = model(*inputs)
        loss += F.cross_entropy(out, y, reduction='sum').item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return correct / n, loss / n


def train_model(model, loaders, epochs=60, lr=1e-3, weight_decay=1e-2, patience=10):
    """
    Trains with AdamW; early stopping and model selection use the val loader only.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training on', device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best_acc, best_state, bad = [], -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        total = n = 0
        for batch in loaders['train']:
            inputs, y = _unpack(batch, device)
            opt.zero_grad()
            loss = F.cross_entropy(model(*inputs), y)
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


BUILDERS = {f.__name__: f for f in (build_mlp, build_net1d, build_simple_cnn,
                                    build_two_branch, build_ts_only, build_grid_only)}

def with_encoder(loaders, encoder, batch_size=128, num_workers=4):
    """Same windows, different image encoder."""
    return {name: DataLoader(l.dataset, batch_size=batch_size, shuffle=(name == 'train'),
                             collate_fn=encoder.collate, num_workers=num_workers,
                             persistent_workers=num_workers > 0)
            for name, l in loaders.items()}

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
    Save contact names and each test/run2 windows the model sees
    """
    d = Path(patient_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'channels.json').write_text(json.dumps(list(channels)))
    arrays = {}
    for split in ('test', 'run2'):
        X, y = loaders[split].dataset.tensors
        arrays[f'{split}_X'] = X.numpy().astype(np.float32)
        arrays[f'{split}_y'] = y.numpy()
    np.savez_compressed(d / 'windows.npz', **arrays)