# MDB-PL

Reference implementation of the model described in the paper.

## Install

```bash
pip install -r requirements.txt
```

## Data

Put the datasets under `data/`: the three JODIE datasets in `data/wikipedia`, `data/reddit`
and `data/mooc`, and the two Bitcoin networks as `data/bitcoinalpha.csv` and
`data/bitcoinotc.csv` with columns `u,i,ts,label,idx`.

Node features are all zero: the model reads the interaction stream, the timestamps and the
structure, not node attributes.

## Run

```bash
python train.py --dataset wikipedia --seed <seed> \
  --aggr gps --hidden_channels 128 --heads 4 --proto_guide sigma_v --assign hard \
  --lr 5e-4 --local_ssm --time_kernel_K 8 --kernel_scale 5.0 --dropout 0.5 \
  --sample_strategy recent --num_neighbors 15 10 --epochs 20 --proto_sep_margin 0.3

python train.py --dataset reddit --seed <seed> \
  --aggr gps --local_ssm --ssm_log_space --no_global_attn --proto_guide cv --assign hard \
  --sample_strategy random --zero_edge --num_neighbors 30 15 --hidden_channels 128 \
  --heads 4 --dropout 0.5 --lr 5e-4 --epochs 20 --proto_sep_margin 0.3

python train.py --dataset mooc --seed <seed> \
  --aggr gps --local_ssm --proto_guide src_deg --assign hard --mem_from_out \
  --sample_strategy recent --num_neighbors 15 10 --hidden_channels 128 --heads 4 \
  --dropout 0.5 --lr 5e-4 --epochs 20 --proto_sep_margin 0.3

python train.py --dataset alpha --seed <seed> \
  --aggr gps --hidden_channels 128 --heads 4 --proto_guide sigma_v --assign hard --lr 2e-4 \
  --temporal_global_kernel --time_kernel_K 4 --kernel_scale 1.0 --dropout 0.5 \
  --num_neighbors 15 10 --epochs 20 --proto_sep_margin 0.3

python train.py --dataset otc --seed <seed> \
  --aggr gps --hidden_channels 128 --heads 4 --proto_guide sigma_v --assign hard --lr 2e-4 \
  --temporal_global_kernel --time_kernel_K 4 --kernel_scale 1.0 --dropout 0.5 \
  --num_neighbors 15 10 --epochs 20 --proto_sep_margin 0.3
```

## Signal selection

`select_signal.py` reports the label-free readings used to choose the behavioural signal of a
dataset: the partial rank correlations between the candidate descriptors and the score of a
light contrastive base, and the degree certificate.

```bash
python select_signal.py --dataset otc
python select_signal.py --dataset reddit --zero_edge
```

## Protocol

Chronological 70/15/15 split at timestamp quantiles. Event-level ROC-AUC, scored on the
source node of each test event at its timestamp. The epoch is chosen by validation AUC.
