# Dynamic graph anomaly detection

Reference implementation of the model described in the submitted manuscript.

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
python train.py --dataset wikipedia --seed 1
python train.py --dataset reddit    --seed 1
python train.py --dataset mooc      --seed 1
python train.py --dataset alpha     --seed 1
python train.py --dataset otc       --seed 1
```

The configuration of each dataset is set in `DATASET_PRESETS` in `train.py` and every option
can be overridden on the command line, for example `--epochs 30` or `--gpu 1`.

## Protocol

Chronological 70/15/15 split at timestamp quantiles. Event-level ROC-AUC, scored on the
source node of each test event at its timestamp. The epoch is chosen by validation AUC.
