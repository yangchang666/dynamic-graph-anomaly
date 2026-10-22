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
python train.py --dataset alpha --seed 1
```

The five datasets are `wikipedia`, `reddit`, `mooc`, `alpha` and `otc`. The main
hyperparameters of each one are listed below; they are set in `DATASET_PRESETS` in `train.py`
and any of them can be overridden on the command line, for example

```bash
python train.py --dataset alpha --seed 1 --epochs 30 --lr 1e-4 --gpu 1
```

| dataset | lr | hidden | heads | dropout | neighbours | epochs |
| --- | --- | --- | --- | --- | --- | --- |
| wikipedia | 5e-4 | 128 | 4 | 0.5 | 15 / 10 | 20 |
| reddit | 5e-4 | 128 | 4 | 0.5 | 30 / 15 | 20 |
| mooc | 5e-4 | 128 | 4 | 0.5 | 15 / 10 | 20 |
| alpha | 2e-4 | 128 | 4 | 0.5 | 15 / 10 | 20 |
| otc | 2e-4 | 128 | 4 | 0.5 | 15 / 10 | 20 |

Batch size is 256, the encoder is the GPS stack and the prototype assignment uses the hard
split with a separation margin of 0.3.

## Protocol

Chronological 70/15/15 split at timestamp quantiles. Event-level ROC-AUC, scored on the
source node of each test event at its timestamp. The epoch is chosen by validation AUC.
