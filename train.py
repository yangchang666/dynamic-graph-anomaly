"""Training and evaluation entry point of MDB-PL.

One run of this script trains the model on one dataset with one seed and prints the
validation and test AUC of the scoring forms at every epoch. The epoch reported is the one
with the best validation AUC of the deployed form of that dataset.
"""
import argparse
import logging
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from core.dataset import load_dataset
from core.history import History
from core.loader import EventLoader
from core.measure import Measure
from core.model import StdGPSModel, SSMGPSModel, GatedTGAT
from core.loss import contrastive_loss, cosine_similarity
from core.descriptors import compute_all_signals, build_guide_signal, DEFAULT_GUIDE


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EMA:
    """参数影子滑动平均; swap 进出用于 eval. 部署形态 = 最终交出 shadow 权重."""

    def __init__(self, modules, decay):
        self.decay = decay
        self.modules = modules
        self.shadow = [{n: p.data.clone() for n, p in m.named_parameters()} for m in modules]

    @torch.no_grad()
    def update(self):
        for sh, m in zip(self.shadow, self.modules):
            for n, p in m.named_parameters():
                if p.numel() == 0 or sh[n].shape != p.data.shape:
                    sh[n] = p.data.clone()  # 零尺寸/动态尺寸参数直接同步
                    continue
                sh[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def swap(self):
        for sh, m in zip(self.shadow, self.modules):
            for n, p in m.named_parameters():
                tmp = p.data.clone()
                p.data.copy_(sh[n])
                sh[n].copy_(tmp)


def train(loader, model, history, optimizer, device, data, args, guide_signal=None, ema=None):
    model.train()
    history.train()
    pbar = tqdm(loader)
    total_loss = 0
    total_proto_loss = 0
    num_batches_with_proto = 0
    params = list(model.parameters()) + list(history.parameters())

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        root_indices = batch.src[:batch.batch_size]
        if guide_signal is not None:
            gv = guide_signal[batch.input_id]
            node_std = torch.zeros(batch.x.size(0), 1, device=device)
            node_std[root_indices] = gv.unsqueeze(1)
            g_assign = gv
        else:
            node_std = torch.zeros(batch.x.size(0), 1, device=device)
            node_std[root_indices] = batch.hop1_time_std
            g_assign = batch.hop1_time_std

        out = model.encode(batch.x, batch.edge_index, batch.t, batch.msg,
                           node_std=node_std)
        src = batch.src[:batch.batch_size]
        out = out[src]
        idx = batch.input_id.cpu()
        num_neg = batch.batch_size
        neg_mem = history.get_history(torch.randint(
            0, history.num_nodes, size=(num_neg,)))
        if args.mem_from_out:
            raw_msg = out
        elif args.real_history:
            raw_msg = data.msg[idx].to(device)
        else:
            raw_msg = torch.zeros(idx.size(0), 1).to(device)
        cur_mem, prev_mem = history(raw_msg, data.src[idx])
        mem = prev_mem if args.mem_from_out else cur_mem

        if args.con == 'his':
            loss = contrastive_loss(prev_mem, cur_mem, args.tau)
            loss += torch.exp(cosine_similarity(cur_mem, neg_mem)
                              ).sum(dim=1).log().mean()
        elif args.con == 'stru':
            loss = contrastive_loss(out, mem, args.tau)
            loss += torch.exp(cosine_similarity(out, neg_mem)
                              ).sum(dim=1).log().mean()
        elif args.con == 'all':
            loss = args.alpha * contrastive_loss(out, mem, args.tau) + \
                contrastive_loss(prev_mem, cur_mem, args.tau)
            loss += torch.exp(cosine_similarity(cur_mem, neg_mem)
                              ).sum(dim=1).log().mean()
            loss += args.alpha * torch.exp(cosine_similarity(out, neg_mem)
                                           ).sum(dim=1).log().mean()

            loss_proto = torch.tensor(0.0, device=device)
            proto_active = False

            if model.use_std and model.std_prototypes is not None:
                proto_src = out if args.proto_train_encoder else out.detach()
                proto_out = model.get_proto_embedding(proto_src)
                g = g_assign.reshape(-1).float()
                sim_high = F.cosine_similarity(
                    proto_out, model.std_prototypes['high'].unsqueeze(0))
                sim_low = F.cosine_similarity(
                    proto_out, model.std_prototypes['low'].unsqueeze(0))
                proto_sep = F.cosine_similarity(
                    model.std_prototypes['high'].unsqueeze(0),
                    model.std_prototypes['low'].unsqueeze(0))

                if args.assign == 'hard':
                    q_low = torch.quantile(g.float(), 0.2)
                    q_high = torch.quantile(g.float(), 0.8)
                    high_mask = (g > q_high)
                    low_mask = (g < q_low)
                    if high_mask.sum() > 0 and low_mask.sum() > 0:
                        loss_high = -sim_high[high_mask].mean()
                        loss_low = -sim_low[low_mask].mean()
                        if args.proto_sep_margin >= 0:
                            loss_proto = (loss_high + loss_low) / 2.0 + (proto_sep.mean() + args.proto_sep_margin).clamp(min=0)
                        else:  # <0 = 旧无界 loss
                            loss_proto = (loss_high + loss_low) / 2.0 + proto_sep.mean()
                        proto_active = True
                else:
                    w_high = torch.sigmoid(g)
                    w_low = 1.0 - w_high
                    loss_high = -(w_high * sim_high).sum() / (w_high.sum() + 1e-8)
                    loss_low = -(w_low * sim_low).sum() / (w_low.sum() + 1e-8)
                    if args.proto_sep_margin >= 0:
                        loss_proto = (loss_high + loss_low) / 2.0 + (proto_sep.mean() + args.proto_sep_margin).clamp(min=0)
                    else:  # <0 = 旧无界 loss
                        loss_proto = (loss_high + loss_low) / 2.0 + proto_sep.mean()
                    proto_active = True

            lambda_proto = args.lambda_proto
            if proto_active:
                loss += lambda_proto * loss_proto
                total_proto_loss += loss_proto.item()
                num_batches_with_proto += 1

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update()
        total_loss += loss.item()
        pbar.set_description(f"Train Loss = {loss.item():.4f}")

    avg_proto_loss = total_proto_loss / max(num_batches_with_proto, 1)
    avg_total_loss = total_loss / len(loader)
    if num_batches_with_proto > 0:
        with torch.no_grad():
            proto_sim = F.cosine_similarity(
                model.std_prototypes['high'].unsqueeze(0),
                model.std_prototypes['low'].unsqueeze(0)).item()
        print(f"[Diag] Proto Loss Contribution: {(avg_proto_loss * args.lambda_proto) / avg_total_loss * 100:.2f}% "
              f"| Proto Similarity: {proto_sim:.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def test(loader, model, history, measure, device, data, args, guide_signal=None, want_loss=False,
         update=True, rep_buf=None):
    """main_seed.test 的复刻 + 逐节点原始分数返回 (+可选 val 对比损失, 无标签)."""
    bases, protos, labels, sim_highs, sim_lows = [], [], [], [], []
    loss_sum, loss_n = 0.0, 0
    model.eval()
    history.eval()

    rep_seen, rep_stride = 0, None
    for batch in tqdm(loader):
        torch.cuda.empty_cache()
        batch = batch.to(device)

        root_indices = batch.src[:batch.batch_size]
        if guide_signal is not None:
            gv = guide_signal[batch.input_id]
            node_std = torch.zeros(batch.x.size(0), 1, device=device)
            node_std[root_indices] = gv.unsqueeze(1)
        else:
            node_std = torch.zeros(batch.x.size(0), 1, device=device)
            node_std[root_indices] = batch.hop1_time_std

        out = model.encode(batch.x, batch.edge_index, batch.t, batch.msg,
                           node_std=node_std)
        src = batch.src[:batch.batch_size]
        out = out[src]
        if rep_buf is not None:
            if rep_stride is None:
                rep_stride = max(1, len(loader.dataset) // max(1, args.dump_rep_n))
            sel = ((torch.arange(out.size(0), device=device) + rep_seen) % rep_stride) == 0
            rep_buf[0].append(out[sel].detach().cpu())
            rep_buf[1].append(model.get_proto_embedding(out)[sel].detach().cpu())
            rep_buf[2].append((torch.arange(out.size(0)) + rep_seen)[sel.cpu()])
            rep_seen += out.size(0)
        label = batch.y[:batch.batch_size]
        idx = batch.input_id.cpu()
        if args.mem_from_out:
            raw_msg = out
        elif args.real_history:
            raw_msg = data.msg[idx].to(device)
        else:
            raw_msg = torch.zeros(idx.size(0), 1).to(device)
        cur_mem, prev_mem = history(raw_msg, data.src[idx], update=update)
        mem = prev_mem if args.mem_from_out else cur_mem

        if want_loss:
            loss_sum += (args.alpha * contrastive_loss(out, mem, args.tau)
                         + contrastive_loss(prev_mem, cur_mem, args.tau)).item()
            loss_n += 1

        p1 = torch.diag(cosine_similarity(out, mem)).view(-1)
        p2 = torch.diag(cosine_similarity(cur_mem, prev_mem)).view(-1)
        base = (1 - p1) + (1 - p2)
        proto_pred = torch.zeros_like(base)
        sh_all = torch.zeros_like(base)
        sl_all = torch.zeros_like(base)

        if model.use_std and model.std_prototypes is not None:
            proto_out = model.get_proto_embedding(out)
            sim_to_high = F.cosine_similarity(
                proto_out, model.std_prototypes['high'].unsqueeze(0), dim=-1)
            sim_to_low = F.cosine_similarity(
                proto_out, model.std_prototypes['low'].unsqueeze(0), dim=-1)
            proto_pred = sim_to_low - sim_to_high
            sh_all, sl_all = sim_to_high, sim_to_low

        bases.append(base.detach().cpu())
        protos.append(proto_pred.detach().cpu())
        sim_highs.append(sh_all.detach().cpu())
        sim_lows.append(sl_all.detach().cpu())
        labels.append(label.detach().cpu())

    base = torch.cat(bases); base[torch.isnan(base)] = 0.0
    proto = torch.cat(protos); proto[torch.isnan(proto)] = 0.0
    sh = torch.cat(sim_highs); sl = torch.cat(sim_lows)
    labels = torch.cat(labels)

    bn = _rank_normalize(base)
    pn = _rank_normalize(proto)
    cand = {
        'add+': (base + proto).sigmoid(),
        'softor+': 1 - (1 - bn) * (1 - pn),
        'max+': torch.max(bn, pn),
        'base': bn,
        'proto+': pn,
    }
    known = (labels != 2)
    aucs = {k: measure(labels[known], v[known]) for k, v in cand.items()}
    vloss = loss_sum / max(loss_n, 1) if want_loss else None
    return aucs, base, proto, sh, sl, labels, vloss


def _rank_normalize(x):
    x = x.float()
    order = x.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(x), dtype=torch.float) + 1.0
    return ranks / len(x)


def _auc(y, s):
    k = y != 2
    return roc_auc_score(y[k].numpy(), s[k].numpy()) if k.sum() > 0 else float('nan')


def read_parser():
    parser = argparse.ArgumentParser(description="main_fix: fusion exploration entry")
    parser.add_argument("--dataset", type=str, default='wikipedia')
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--con", type=str, default='all')
    parser.add_argument("--aggr", type=str, default='gps')
    parser.add_argument("--num_neighbors", nargs="+", type=int)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_timeslots", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--nums_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--history_retrieve", type=str, default='mean')
    parser.add_argument("--recurrent", type=str, default='gru')
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_strategy", type=str, default="recent",
                        choices=["recent", "random"])
    parser.add_argument("--zero_edge", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--proto_guide", type=str, default="sigma_v")
    parser.add_argument("--assign", type=str, default="hard", choices=["soft", "hard"])
    parser.add_argument("--local_ssm", action="store_true", default=False)
    parser.add_argument("--ssm_log_space", action="store_true", default=False)
    parser.add_argument("--no_global_attn", action="store_true", default=False)
    parser.add_argument("--temporal_global_kernel", action="store_true", default=False)
    parser.add_argument("--time_kernel_K", type=int, default=8)
    parser.add_argument("--kernel_scale", type=float, default=5.0)
    parser.add_argument("--gps_local_type", type=str, default='GINEConv')
    parser.add_argument("--proto_train_encoder", action="store_true", default=False)
    parser.add_argument("--no_std", action="store_true", default=False,
                        help="去掉 use_std 整条支路(统计量注入 + 原型分支), 退化为纯对比损失")
    parser.add_argument("--dump_rep", action="store_true", default=False,
                        help="每个 epoch 等距抽样存根节点表示(out)与原型头表示, 供表示可分性分析")
    parser.add_argument("--dump_rep_n", type=int, default=3000,
                        help="--dump_rep 的抽样事件数")
    parser.add_argument("--lambda_proto", type=float, default=1.0)
    parser.add_argument("--real_history", action="store_true", default=False)
    parser.add_argument("--mem_from_out", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proto_sep_margin", type=float, default=-1.0,
                        help="prototype separation margin; -1.0 disables it")
    # ===== v2 稳定化仪器 (默认全关 = 原行为) =====
    parser.add_argument("--ema", action="store_true", default=False,
                        help="权重 EMA, eval/部署用 shadow 权重")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--cosine_lr", action="store_true", default=False,
                        help="余弦退火 lr -> 0")
    parser.add_argument("--wd", type=float, default=0.0,
                        help=">0 时 AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help=">0 时梯度范数裁剪")
    parser.add_argument("--val_loss", action="store_true", default=False,
                        help="记录 val 对比损失 (无标签)")
    parser.add_argument("--out_tag", type=str, default="",
                        help="npz 文件名后缀")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = read_parser()
    if not os.path.exists('log'):
        os.makedirs('log')
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=logging.INFO,
                        filename=f'log/{osp.basename(__file__)[:-3]}_{args.dataset}.txt')
    log = logging.getLogger()
    log.info('########################## START #########################')
    log.info(f'\n{args.dataset}-param: {args.__dict__}')

    data = load_dataset(args.dataset, zero_edge=args.zero_edge)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using {device}")

    guide_signal = None
    feat = args.proto_guide
    if feat == "auto":
        feat = DEFAULT_GUIDE.get(args.dataset, "sigma_v")
    if feat != "sigma_v":
        signals = compute_all_signals(data, args.dataset)
        guide_np, guide_auc = build_guide_signal(data, signals, feat)
        guide_signal = torch.from_numpy(guide_np).to(device)
        print(f"[guide] dataset={args.dataset} feature={feat} trainAUC={guide_auc:.3f}")
    else:
        print(f"[guide] dataset={args.dataset} feature=sigma_v (batch.hop1_time_std)")

    set_seed(args.seed)
    sample_args = {"num_neighbors": args.num_neighbors,
                   "num_workers": args.num_workers,
                   "sample_strategy": args.sample_strategy}
    train_loader = EventLoader(data, input_events=data.train_mask, shuffle=True,
                               batch_size=args.batch_size, **sample_args)
    valid_loader = EventLoader(data, input_events=data.val_mask, shuffle=False,
                               batch_size=args.batch_size, **sample_args)
    test_loader = EventLoader(data, input_events=data.test_mask, shuffle=False,
                              batch_size=args.batch_size, **sample_args)

    history = History(data.num_nodes, args.num_timeslots, args.hidden_channels,
                      device=device, history_retrieve=args.history_retrieve,
                      recurrent=args.recurrent).to(device)
    if args.aggr == 'gps':
        gps_kw = dict(
            in_channels=data.x.size(1), out_channels=1,
            hidden_channels=args.hidden_channels, edge_dim=data.msg.size(1),
            num_layers=args.nums_layers, dropout=args.dropout,
            use_std=not args.no_std, use_single_proto=False,
            gps_heads=args.heads, gps_dropout=args.dropout,
            temporal_global_kernel=args.temporal_global_kernel,
            time_kernel_K=args.time_kernel_K, time_kernel_scale=args.kernel_scale,
        )
        if args.local_ssm:
            model = SSMGPSModel(**gps_kw, no_global_attn=args.no_global_attn,
                                log_space=args.ssm_log_space).to(device)
        else:
            model = StdGPSModel(**gps_kw, gps_local_type=args.gps_local_type).to(device)
    elif args.aggr == 'gat':
        # encoder-replacement ablation: swap in the GatedTGAT encoder, keep everything else.
        # gps 专属参数在此分支不适用, 且 gat 无全局注意力。
        model = GatedTGAT(
            in_channels=data.x.size(1), out_channels=1,
            hidden_channels=args.hidden_channels, edge_dim=data.msg.size(1),
            heads=args.heads, num_layers=args.nums_layers, dropout=args.dropout,
            use_std=not args.no_std, use_single_proto=False,
        ).to(device)
    else:
        raise ValueError("main_fix 只支持 --aggr gps / gat")
    print(model)
    print(history)

    if args.wd > 0:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(history.parameters()), lr=args.lr,
            weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(history.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs) if args.cosine_lr else None
    ema = EMA([model, history], args.ema_decay) if args.ema else None
    measure = Measure("auc")

    store = {}
    for epoch in range(1, 1 + args.epochs):
        loss = train(train_loader, model, history, optimizer, device, data, args,
                     guide_signal=guide_signal, ema=ema)
        if scheduler is not None:
            scheduler.step()
        # 1) 原始权重评估 (update=True, 与既有评估行为一致)
        v_out = test(valid_loader, model, history, measure, device, data, args,
                     guide_signal=guide_signal, want_loss=args.val_loss)
        v_aucs, v_base, v_proto, v_sh, v_sl, v_y, v_loss = v_out
        rep_buf = [[], [], []] if args.dump_rep else None
        t_aucs, t_base, t_proto, t_sh, t_sl, t_y, _ = test(
            test_loader, model, history, measure, device, data, args, guide_signal=guide_signal,
            rep_buf=rep_buf)
        if rep_buf is not None:
            store[f"t_rep_e{epoch}"] = torch.cat(rep_buf[0]).numpy().astype(np.float32)
            store[f"t_prep_e{epoch}"] = torch.cat(rep_buf[1]).numpy().astype(np.float32)
            if "t_rep_idx" not in store:
                store["t_rep_idx"] = torch.cat(rep_buf[2]).numpy().astype(np.int32)
        # 2) EMA 同跑配对: swap 后 update=False (history 冻结 -> 隔离权重效应, 不扰动训练)
        if ema is not None:
            ema.swap()
            _, ev_base, ev_proto, ev_sh, ev_sl, _, _ = test(
                valid_loader, model, history, measure, device, data, args,
                guide_signal=guide_signal, update=False)
            _, et_base, et_proto, et_sh, et_sl, _, _ = test(
                test_loader, model, history, measure, device, data, args,
                guide_signal=guide_signal, update=False)
            ema.swap()
            store[f"ve_base_e{epoch}"], store[f"ve_proto_e{epoch}"] = ev_base.numpy(), ev_proto.numpy()
            store[f"ve_sh_e{epoch}"], store[f"ve_sl_e{epoch}"] = ev_sh.numpy(), ev_sl.numpy()
            store[f"te_base_e{epoch}"], store[f"te_proto_e{epoch}"] = et_base.numpy(), et_proto.numpy()
            store[f"te_sh_e{epoch}"], store[f"te_sl_e{epoch}"] = et_sh.numpy(), et_sl.numpy()
            print(f"  [EMA] test base={_auc(t_y, et_base):.4f} proto={_auc(t_y, _rank_normalize(et_sl - et_sh)):.4f}", flush=True)

        store[f"v_base_e{epoch}"], store[f"v_proto_e{epoch}"] = v_base.numpy(), v_proto.numpy()
        store[f"v_sh_e{epoch}"], store[f"v_sl_e{epoch}"] = v_sh.numpy(), v_sl.numpy()
        store["v_y"] = v_y.numpy()
        store[f"t_base_e{epoch}"], store[f"t_proto_e{epoch}"] = t_base.numpy(), t_proto.numpy()
        store[f"t_sh_e{epoch}"], store[f"t_sl_e{epoch}"] = t_sh.numpy(), t_sl.numpy()
        store["t_y"] = t_y.numpy()
        if v_loss is not None:
            store[f"v_loss_e{epoch}"] = v_loss

        best_v = max(v_aucs, key=v_aucs.get)
        vlp = f" vloss={v_loss:.3f}" if v_loss is not None else ""
        print(f"Epoch {epoch:03d} loss={loss:.4f}{vlp} | val best={best_v}:{v_aucs[best_v]:.4f} "
              f"(base {v_aucs['base']:.4f} proto {v_aucs['proto+']:.4f}) | "
              f"test base={t_aucs['base']:.4f} proto={t_aucs['proto+']:.4f} max={t_aucs['max+']:.4f}",
              flush=True)
        log.info(f"Epoch {epoch:03d} loss={loss:.4f}{vlp} val_best={best_v}:{v_aucs[best_v]:.4f} "
                 f"test base={t_aucs['base']:.4f} proto={t_aucs['proto+']:.4f}")

    tag = f"_{args.out_tag}" if args.out_tag else ""
    npz_path = f"fusion_{args.dataset}_s{args.seed}{tag}.npz"
    np.savez_compressed(npz_path, **store)
    print("saved:", npz_path)

    # ===== 融合对照: 各变体各自 val 最优 epoch 的 test AUC + 天花板 =====
    variants = {}
    for ep in range(1, 1 + args.epochs):
        vb = torch.from_numpy(store[f"v_base_e{ep}"]); vp = torch.from_numpy(store[f"v_proto_e{ep}"])
        vsh = torch.from_numpy(store[f"v_sh_e{ep}"]); vsl = torch.from_numpy(store[f"v_sl_e{ep}"])
        vy = torch.from_numpy(store["v_y"])
        tb = torch.from_numpy(store[f"t_base_e{ep}"]); tp = torch.from_numpy(store[f"t_proto_e{ep}"])
        tsh = torch.from_numpy(store[f"t_sh_e{ep}"]); tsl = torch.from_numpy(store[f"t_sl_e{ep}"])
        ty = torch.from_numpy(store["t_y"])
        bn, pn = _rank_normalize(vb), _rank_normalize(vp)
        tbn, tpn = _rank_normalize(tb), _rank_normalize(tp)
        c = _rank_normalize((vsh - vsl).abs())
        tc = _rank_normalize((tsh - tsl).abs())
        for name, vs, ts in [
            ("proto-only", pn, tpn),
            ("base-only", bn, tbn),
            ("max(bn,pn)", torch.max(bn, pn), torch.max(tbn, tpn)),
        ]:
            variants.setdefault(name, []).append((_auc(vy, vs), _auc(ty, ts)))

    print(f"\n===== {args.dataset} seed {args.seed}: scoring forms (val-best epoch -> test AUC) =====")
    for name, rows in variants.items():
        bep = int(np.argmax([r[0] for r in rows])) + 1
        cep = int(np.argmax([r[1] for r in rows])) + 1
        print(f"{name:20s} val-best ep{bep:02d} test={rows[bep-1][1]:.4f} | ceiling ep{cep:02d} test={rows[cep-1][1]:.4f}")
    if f"v_loss_e{args.epochs}" in store:
        vl = [store[f"v_loss_e{e}"] for e in range(1, 1 + args.epochs)]
        lep = int(np.argmin(vl)) + 1
        print(f"[val-loss dial] 最优 ep{lep:02d} (vloss={vl[lep-1]:.3f})")
