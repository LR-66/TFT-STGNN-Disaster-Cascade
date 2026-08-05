# Cross-Regional Hydro-Meteorological Cascade Propagation Modeling and Emergency Response Optimization
### TFT-ST-GNN Joint Framework — Complete Replication Package

This repository implements the complete replication package described in the paper:

> *Cross-Regional Hydro-Meteorological Cascade Propagation Modeling and Emergency
> Response Optimization Using Temporal Fusion Transformer and Spatio-Temporal GNN*
> (Longkai Chen)

It provides, as required by the paper's *Availability of Data and Materials* section:

1. A **multi-agent simulation framework** calibrated with EM-DAT, OSM, WorldPop,
   SRTM and LISFLOOD-style physical rules (see `src/simulation/`).
2. All **data preprocessing and standardization scripts** (detrending,
   log-stabilization, sliding-window standardization, Z-score by region type;
   see `src/preprocessing/`).
3. **Dynamic graph construction and adaptive adjacency learning** code
   (distance-decay prior + learnable adjacency; see `src/models/stgnn.py`).
4. **PyTorch implementation of the TFT-ST-GNN model** with the full
   hyperparameter configuration of the paper (see `config/config.py`).
5. Generated **training / validation / test datasets in CSV format**
   (written to `data/`).
6. **Integer-programming formulation** of the emergency resource scheduling
   with probabilistic (quantile) constraints. The optimization model is exported
   in **LP format** (see `src/optimization/lp_writer.py`). The solver uses
   **Gurobi** when `gurobipy` is installed, and otherwise falls back to a
   built-in deterministic priority solver so the pipeline runs everywhere.
7. **Configuration scripts for all baseline models** (PatchTST, Informer,
   DCRNN, Graph WaveNet, AGCRN; see `src/baselines/`).
8. **Trained model weight files** saved under `outputs/models/` after training.

All random seeds of the paper are respected: `42` (data generation),
`123` (train/validation/test split), `456` (model initialization),
`789` (baseline reproducibility), `101112` (additional runs).

---

## Directory Layout

```
cascade_emergency_response/
├── README.md
├── requirements.txt
├── run_all.py                  # one-click end-to-end pipeline (PyCharm: run this)
├── config/
│   └── config.py               # ALL hyperparameters of the paper
├── src/
│   ├── simulation/             # (1) multi-agent cascade simulation
│   │   ├── __init__.py
│   │   ├── static_attributes.py
│   │   ├── hydrology.py
│   │   ├── graph_builder.py
│   │   ├── cascade_env.py
│   │   └── data_generator.py
│   ├── preprocessing/          # (2) transforms + PyTorch datasets
│   │   ├── __init__.py
│   │   ├── transforms.py
│   │   └── dataset.py
│   ├── models/                 # (4) TFT-ST-GNN joint model
│   │   ├── __init__.py
│   │   ├── tft.py
│   │   ├── stgnn.py
│   │   └── tft_stgnn.py
│   ├── baselines/              # (7) PatchTST / Informer / DCRNN / GraphWaveNet / AGCRN
│   │   ├── __init__.py
│   │   ├── patchtst.py
│   │   ├── informer.py
│   │   ├── dcrnn.py
│   │   ├── graphwavenet.py
│   │   └── agcrn.py
│   ├── optimization/           # (6) integer programming scheduling
│   │   ├── __init__.py
│   │   ├── lp_writer.py
│   │   ├── emergency_optimizer.py
│   │   └── dispatch_sim.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # RMSE/MAE/PICP/PINAW/CRPS/Winkler/Recall/Precision/F1
│   └── utils/
│       ├── __init__.py
│       └── seed.py
├── scripts/                    # step-by-step entry points
│   ├── 01_generate_data.py
│   ├── 02_train_tft_stgnn.py
│   ├── 03_evaluate.py
│   ├── 04_optimize_schedule.py
│   ├── 05_run_baselines.py
│   └── 06_ablation.py
├── data/                       # generated CSV datasets (auto-created)
└── outputs/                    # models / metrics / schedules / lp files (auto-created)
```

## Quick Start (PyCharm)

1. Open this folder as a project in PyCharm.
2. Create a virtual environment (Python >= 3.8 recommended; 3.6 also works) and
   install dependencies: `pip install -r requirements.txt`
   (only `numpy`, `torch`, `matplotlib` are strictly required; `gurobipy`,
   `pandas`, `scipy` are optional enhancements).
3. Run `run_all.py` (or run the numbered scripts in `scripts/` in order).

`run_all.py` runs the full pipeline: data generation → preprocessing → training →
evaluation → emergency scheduling. After it finishes you will find:

- `data/`: `covariates.csv`, `targets.csv`, `static_attributes.csv`,
  `adjacency.csv`, `train.csv`, `val.csv`, `test.csv`
- `outputs/models/`: `tft_stgnn.pt` (trained weights)
- `outputs/results/`: `metrics.json` (RMSE, MAE, PICP, PINAW, CRPS, Winkler,
  recall, precision, F1 — compared against the values reported in the paper)
- `outputs/schedules/`: dispatch plans + `.lp` optimization model files

## Reproducing the Paper's Reported Numbers

The evaluation script prints every metric of the paper (Section 4) together with
the value obtained by your run:

| Metric (paper, Section 4)              | Reported value |
|----------------------------------------|----------------|
| RMSE (average, high-disturbance)       | ~0.071         |
| Recall (extreme nodes, dense topology) | ~0.97          |
| Precision (extreme nodes)              | ~0.95          |
| F1 score                              | ~0.96          |
| PICP (80% interval)                    | ~0.795         |
| PINAW                                 | ~0.182         |
| CRPS                                  | ~0.068         |
| Winkler                               | ~0.241         |
| Response time @ 500 nodes              | ~5.6 s         |

To get close to these numbers run the **full configuration**
(`python run_all.py --nodes 50 --hours 10000 --epochs 100`); a GPU is not
required but speeds training up significantly. The small default of
`run_all.py` is a smoke test that completes in minutes on CPU.

## Notes

- The Gurobi solver (paper: Gurobi 9.5, 60 s time limit) is used automatically
  when `gurobipy` is importable; otherwise a deterministic built-in
  priority/greedy solver produces the same plan format so the pipeline always
  runs. The LP model files are always exported regardless of solver.
- Baselines are faithful but compact implementations of the paper's
  configurations (see `config/config.py` and `src/baselines/`).
- Everything is deterministic given the seeds above.
