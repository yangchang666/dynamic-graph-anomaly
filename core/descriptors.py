"""Behavioural descriptors and construction of the prototype guiding signal.

Descriptors (one value per event, causal, in the original event order):
    sigma_v      standard deviation of the first-hop gaps
    src_deg      cumulative number of events of the source node
    dst_deg      cumulative number of events of the destination node
    src_recency  time since the previous interaction of the source node
    int_mean     mean historical gap of the source node
    int_std      standard deviation of the historical gaps of the source node
    cv           coefficient of variation, std / mean
    burstiness   (mean - std) / (mean + std)

The guiding signal g is a scalar per event and is oriented so that the low end is the
anomalous one: the selected descriptor is z-scored with training statistics and its sign is
flipped when the training AUC is below 0.5. The low prototype then plays the role of the
anomaly anchor, and the score cos(h, p_low) keeps the orientation used by the model.

The default guiding signal of a dataset is fixed by the selection procedure of Section 3.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_GUIDE = {
    "wikipedia": "sigma_v",
    "reddit": "sigma_v",
    "mooc": "src_deg",
    "amazon": "dst_deg",
    "alpha": "src_deg",
    "otc": "src_deg",
}


def compute_temporal(src, dst, t):
    """Causal temporal and structural descriptors, one value per event."""
    order = np.argsort(t, kind="stable")
    back = np.argsort(order, kind="stable")
    df = pd.DataFrame({"src": src[order], "dst": dst[order],
                       "t": t[order].astype(np.float64)})
    df["gap"] = df.groupby("src")["t"].diff()
    grp = df.groupby("src")["gap"]
    df["int_mean"] = grp.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    df["int_std"] = grp.expanding(min_periods=1).std().reset_index(level=0, drop=True)
    df["src_deg"] = df.groupby("src").cumcount().astype(np.float32)
    df["dst_deg"] = df.groupby("dst").cumcount().astype(np.float32)
    df["src_recency"] = df["gap"]
    mu, sd = df["int_mean"], df["int_std"]
    df["cv"] = sd / (mu.abs() + 1e-9)
    df["burstiness"] = (mu - sd) / (mu + sd + 1e-9)
    out = {}
    for c in ["src_deg", "dst_deg", "src_recency", "int_mean", "int_std",
              "cv", "burstiness"]:
        v = df[c].values.astype(np.float32)
        v[~np.isfinite(v)] = 0.0
        out[c] = v[back]
    return out


def compute_all_signals(data, name):
    """Descriptor dictionary for one dataset, cached in data/{name}_signals.npz."""
    cache = f"data/{name}_signals.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    print(f"[descriptors] computing signals for {name} ...")
    src = data.src.numpy(); dst = data.dst.numpy(); t = data.t.numpy()
    out = compute_temporal(src, dst, t)
    np.savez(cache, **out)
    print(f"[descriptors] cached -> {cache}")
    return out


def build_guide_signal(data, signals, feature):
    """Selected descriptor, z-scored with training statistics and oriented so that the
    low end is the anomalous one."""
    raw = signals[feature].astype(np.float64)
    tr = data.train_mask.numpy()
    y = data.y.numpy().astype(int)
    known_tr = tr & (y != 2)

    mu = raw[tr].mean()
    sd = raw[tr].std() + 1e-9
    z = (raw - mu) / sd

    if known_tr.sum() > 0 and len(np.unique(y[known_tr])) == 2:
        auc = roc_auc_score(y[known_tr], z[known_tr])
        if auc < 0.5:
            z = -z
            auc = 1 - auc
    else:
        auc = float("nan")
    z = np.nan_to_num(z.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[descriptors] guide='{feature}'  trainAUC(best-dir)={fmt(auc)}  "
          f"(low = anomalous, standardised on train)")
    return z, auc


def fmt(x, p=3):
    return "  -  " if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{p}f}"


def build_multi_guide(data, name, num_neighbors):
    """Guiding matrix with one column per descriptor, z-scored on the training events.
    sigma_v is sampled through the same first-hop path as the model.
    """
    from find_proto_signal import extract_std
    signals = compute_all_signals(data, name)
    sigma_v = extract_std(data, num_neighbors)
    raw = np.stack([
        signals["src_deg"].astype(np.float64),
        sigma_v.astype(np.float64),
        signals["cv"].astype(np.float64),
    ], axis=1)                                            # [N, 3]
    tr = data.train_mask.numpy()
    mu = raw[tr].mean(axis=0)                             # [3]
    sd = raw[tr].std(axis=0) + 1e-9
    raw = (raw - mu) / sd                                 # train z-score
    raw = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[descriptors] multi_guide: [src_deg, σv, cv] (z-scored on train), shape={raw.shape}")
    return raw
