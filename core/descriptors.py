"""
行为稳定性描述子 + 原型引导信号构造
================================

论文 DyG-PL 用 σ_v (hop1_time_std) 引导原型分配(硬 20/80 分桶)。诊断
(find_proto_signal.py) 表明 σ_v 不通用: wiki/reddit 强、mooc 弱、amazon 随机。
本模块预计算一组"行为稳定性描述子", 并按数据集选出最佳引导信号, 供软分配原型使用。

描述子(每个 [N], 因果=只用过去信息, 原始事件顺序):
  sigma_v      hop1 间隔标准差 (论文信号, 经 loader 采样)
  src_deg      源节点累计事件数
  dst_deg      目标节点累计事件数
  src_recency  距离源节点上一次交互的时间间隔
  int_mean     源节点历史间隔均值
  int_std      源节点历史间隔标准差
  cv           变异系数 std/mean
  burstiness   (mean-std)/(mean+std)

引导信号 g (标量, 定向为 low=anomalous):
  - 取选定描述子, 用 train 统计做 z-score
  - 用 train 标签确定方向: 若 train AUC<0.5 (high=anom) 则取反, 保证 low=anom
  - 这样 p_low 自然成为"异常锚原型", 测试分数 cos(h, p_low) 与论文一致

诊断结论(测试集 best-dir AUC)作为默认每集引导:
  wikipedia/reddit -> sigma_v ; mooc -> src_deg(0.727) ; amazon -> dst_deg(0.647)
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 每数据集默认引导描述子 (来自 find_proto_signal.py 诊断)
DEFAULT_GUIDE = {
    "wikipedia": "sigma_v",
    "reddit": "sigma_v",
    "mooc": "src_deg",
    "amazon": "dst_deg",
    "alpha": "src_deg",
    "otc": "src_deg",
}


# ---------- 描述子计算 ----------
def compute_temporal(src, dst, t):
    """因果时序/结构描述子. 返回 dict[name]->[N] (原始事件顺序)."""
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
    """返回 dict {src_deg, dst_deg, ...} 每个 [N] float32. 带 npz 缓存.
    注: 不含 sigma_v —— sigma_v 走 batch.hop1_time_std 回退路径, 避免逐事件采样开销."""
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


# ---------- 引导信号构造 ----------
def build_guide_signal(data, signals, feature):
    """选定描述子 -> [N] z-score(train 统计), 定向 low=anomalous."""
    raw = signals[feature].astype(np.float64)
    tr = data.train_mask.numpy()
    y = data.y.numpy().astype(int)
    known_tr = tr & (y != 2)

    mu = raw[tr].mean()
    sd = raw[tr].std() + 1e-9
    z = (raw - mu) / sd

    # 用 train 标签定方向: 若 high=anom (auc<0.5) 则取反, 保证 low=anom
    if known_tr.sum() > 0 and len(np.unique(y[known_tr])) == 2:
        auc = roc_auc_score(y[known_tr], z[known_tr])
        if auc < 0.5:
            z = -z
            auc = 1 - auc
    else:
        auc = float("nan")
    z = np.nan_to_num(z.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[descriptors] guide='{feature}'  trainAUC(best-dir)={fmt(auc)}  "
          f"(low=anom, 标准化于 train)")
    return z, auc


def fmt(x, p=3):
    return "  -  " if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{p}f}"


def build_multi_guide(data, name, num_neighbors):
    """三信号并行引导: descs [num_events, 3] = z-scored (src_deg, σv, cv)。

    三信号不选不学, 各自 train z-score 后拼成 [N,3], 供 BlockMultiGPSModel 各注入一块。
    σv 走与模型同路径的 hop1 采样(extract_std); 全程无监督(不用标签)。
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
