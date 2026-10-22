"""Label-free readings behind the choice of the behavioural signal (Section 3).

A light contrastive base, an encoder without the prototype branch, is trained for a fixed
number of epochs and scores the validation events. Four readings are then computed against
those scores:

    p_sv    partial Spearman between sigma_v and the base score, controlling degree
    p_cv    partial Spearman between c_v and the base score, controlling degree
    r_deg   Spearman between degree and the base score
    dR2     increment of R^2 of the base score when the degree block is augmented with the
            temporal descriptors

No label is read anywhere in this script.

Usage:
    python select_signal.py --dataset wikipedia
    python select_signal.py --dataset reddit --zero_edge --epochs 10
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, rankdata, spearmanr
from tqdm import tqdm

from core.dataset import load_dataset
from core.history import History
from core.loader import EventLoader
from core.loss import contrastive_loss, cosine_similarity
from core.model import GatedTGAT

BLK = ["sigma_v", "cv_recent", "cv"]


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rank01(x):
    r = rankdata(np.asarray(x, dtype=np.float64))
    return (r - 1.0) / (len(r) - 1.0)


def r2(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    return 1.0 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def partial_spearman(x, y, z):
    rx, ry, rz = rank01(x), rank01(y), rank01(z)
    Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(pearsonr(ex, ey)[0])


def descriptors(src, dst, t):
    """Degree and the coefficient of variation of the inter-event times, per event."""
    order = np.argsort(t, kind="stable")
    back = np.argsort(order, kind="stable")
    df = pd.DataFrame({"src": src[order], "t": t[order].astype(np.float64)})
    df["gap"] = df.groupby("src")["t"].diff()
    grp = df.groupby("src")["gap"]
    df["mean"] = grp.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    df["std"] = grp.expanding(min_periods=1).std().reset_index(level=0, drop=True)
    df["src_deg"] = df.groupby("src").cumcount().astype(np.float32)
    df["cv"] = (df["std"] / (df["mean"].abs() + 1e-9)).astype(np.float32)
    out = {}
    for c in ["src_deg", "cv"]:
        v = df[c].values.astype(np.float32)
        v[~np.isfinite(v)] = 0.0
        out[c] = v[back]
    return out


def recent_stats(src, t, M):
    """Standard deviation, and its normalised version, of the gaps in a recent window."""
    N = len(src)
    order = np.argsort(t, kind="stable")
    back = np.argsort(order, kind="stable")
    df = pd.DataFrame({"s": src[order], "t": t[order].astype(np.float64)})
    sv = np.zeros(N, dtype=np.float32)
    cvr = np.zeros(N, dtype=np.float32)
    for _, grp in df.groupby("s"):
        ts = grp["t"].values
        pos = grp.index.values
        for p in range(len(ts)):
            window = ts[max(0, p - M + 1):p + 1]
            if len(window) >= 2:
                gaps = np.append(np.diff(window), 0.0)
                sv[pos[p]] = gaps.std()
                cvr[pos[p]] = gaps.std() / (abs(gaps.mean()) + 1e-9)
    return sv[back], cvr[back]


def train(loader, model, history, opt, device, data, args):
    model.train(); history.train()
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        opt.zero_grad()
        out = model.encode(batch.x, batch.edge_index, batch.t, batch.msg)
        out = out[batch.src[:batch.batch_size]]
        idx = batch.input_id.cpu()
        neg_mem = history.get_history(torch.randint(0, history.num_nodes, size=(batch.batch_size,)))
        raw_msg = torch.zeros(idx.size(0), 1).to(device)
        cur_mem, prev_mem = history(raw_msg, data.src[idx])
        loss = args.alpha * contrastive_loss(out, cur_mem, args.tau) \
            + contrastive_loss(prev_mem, cur_mem, args.tau)
        loss = loss + torch.exp(cosine_similarity(cur_mem, neg_mem)).sum(dim=1).log().mean()
        loss = loss + args.alpha * torch.exp(cosine_similarity(out, neg_mem)).sum(dim=1).log().mean()
        loss.backward()
        opt.step()


@torch.no_grad()
def score(loader, model, history, device, data):
    model.eval(); history.eval()
    ids, scs = [], []
    for batch in tqdm(loader, desc="score", leave=False):
        batch = batch.to(device)
        out = model.encode(batch.x, batch.edge_index, batch.t, batch.msg)
        out = out[batch.src[:batch.batch_size]]
        idx = batch.input_id
        raw_msg = torch.zeros(idx.size(0), 1).to(device)
        cur_mem, prev_mem = history(raw_msg, data.src[idx.cpu()], update=True)
        p1 = torch.diag(cosine_similarity(out, cur_mem)).view(-1)
        p2 = torch.diag(cosine_similarity(cur_mem, prev_mem)).view(-1)
        ids.append(idx.cpu()); scs.append(((1 - p1) + (1 - p2)).detach().cpu())
    return torch.cat(ids).numpy(), np.nan_to_num(torch.cat(scs).numpy())


def readings(desc, scs):
    blk = [rank01(desc[k]) for k in BLK]
    deg = rank01(desc["src_deg"])
    return (partial_spearman(desc["sigma_v"], scs, deg),
            partial_spearman(desc["cv"], scs, deg),
            spearmanr(desc["src_deg"], scs)[0],
            r2(rank01(scs), blk + [deg]) - r2(rank01(scs), [deg]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--zero_edge", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 2024])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_neighbors", nargs="+", type=int, default=[15, 10])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()

    data = load_dataset(a.dataset, zero_edge=a.zero_edge)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    src, dst, t = data.src.numpy(), data.dst.numpy(), data.t.numpy()
    desc = descriptors(src, dst, t)
    sv, cvr = recent_stats(src, t, a.num_neighbors[0])
    cobj = argparse.Namespace(alpha=1.0, tau=0.02)

    ids0, all_scs = None, {}
    for seed in a.seeds:
        set_seed(seed)
        loaders = dict(
            train=EventLoader(data, input_events=data.train_mask, shuffle=True,
                              batch_size=a.batch_size, num_neighbors=a.num_neighbors,
                              num_workers=0),
            val=EventLoader(data, input_events=data.val_mask, shuffle=False,
                            batch_size=a.batch_size, num_neighbors=a.num_neighbors,
                            num_workers=0))
        history = History(data.num_nodes, 1, a.hidden, device=device,
                          history_retrieve="mean", recurrent="gru").to(device)
        model = GatedTGAT(in_channels=data.x.size(1), out_channels=1,
                          hidden_channels=a.hidden, edge_dim=data.msg.size(1),
                          heads=1, num_layers=2, dropout=0.5).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(history.parameters()), lr=a.lr)
        for ep in range(1, a.epochs + 1):
            train(loaders["train"], model, history, opt, device, data, cobj)
            ids, scs = score(loaders["val"], model, history, device, data)
            ids0 = ids if ids0 is None else ids0
            all_scs.setdefault(seed, {})[ep] = scs
            print(f"[{a.dataset} seed{seed}] epoch {ep}", flush=True)

    d = {"src_deg": desc["src_deg"][ids0], "sigma_v": sv[ids0],
         "cv": desc["cv"][ids0], "cv_recent": cvr[ids0]}
    print(f"{'seed':>6} {'ep':>3} {'p_sv':>7} {'p_cv':>7} {'r_deg':>7} {'dR2':>6}")
    for seed in a.seeds:
        for ep in range(1, a.epochs + 1):
            out = readings(d, all_scs[seed][ep])
            print(f"{seed:>6} {ep:>3} {out[0]:>+7.3f} {out[1]:>+7.3f} {out[2]:>+7.3f} {out[3]:>6.3f}")


if __name__ == "__main__":
    main()
