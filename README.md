# turbofan-prognostics-mlops

Physics-informed Remaining Useful Life (RUL) prediction for turbofan jet
engines, built on NASA's C-MAPSS dataset, wrapped in a full MLOps loop:
drift detection, automatic retraining, CI/CD, and monitoring. Includes an
agentic layer that cross-references live FAA Airworthiness Directives
against a given engine and drafts compliance-aware maintenance
recommendations.

This is a from-scratch build: every function here was written and verified
against real NASA sensor data, and the commit history reflects the actual
build order, bugs caught along the way included.

## Problem

Airlines and MROs run two things as separate manual processes that should be
connected: engine-health monitoring (is this engine degrading faster than
expected?) and regulatory compliance (does an open Airworthiness Directive
apply to this engine?). Most predictive-maintenance ML in production also
lacks real MLOps discipline — no drift detection, no retraining pipeline —
so models degrade silently over time without anyone noticing.

## What's built so far

- **Data pipeline** (`src/turbofan_prognostics/data.py`): loads and labels
  raw C-MAPSS sensor data, computes RUL (Remaining Useful Life) for both
  training data (run-to-failure) and test data (via NASA's separate
  ground-truth file), with input validation and tests against real data.
- **Engine-level train/validation split** (`splits.py`): splits by engine,
  not by row, to avoid leaking one engine's degradation curve across both
  sets.
- **Feature scaling** (`scaling.py`): z-score normalization fit on training
  data only; automatically detects and drops zero-variance columns (caught
  a real divide-by-zero bug this way — see commit history).
- **Exploratory analysis** (`viz.py`): ranks sensors by correlation with
  RUL, visualizes degradation trends aligned by life-fraction across
  engines.

See `reports/figures/top_sensor_degradation.png`: sensor readings are
flat/noisy early in an engine's life and trend sharply as failure
approaches — the physical signal the model will learn from.

## Roadmap

- [ ] Sliding-window sequence generation for sequence models
- [ ] Baseline RUL regression model
- [ ] Physics-informed hybrid model (adds a degradation-physics loss term)
- [ ] MLflow experiment tracking
- [ ] Evidently drift detection + automatic retraining trigger
- [ ] FastAPI serving + Docker + CI/CD (GitHub Actions)
- [ ] Prometheus/Grafana monitoring
- [ ] LangGraph agent for FAA Airworthiness Directive cross-referencing
- [ ] n8n approval routing

## Dataset

NASA Turbofan Engine Degradation Simulation (C-MAPSS): 4 sub-datasets
(FD001-FD004) of simulated multivariate sensor time series for fleets of
turbofan engines run to failure under varying operating conditions and
fault modes. FD001 (single operating condition, single fault mode) is the
current working set; FD002-FD004 add multiple operating conditions and
fault modes. Not committed to git — see `data/raw/` (gitignored).

Re-fetch it with:

    mkdir -p data/raw && cd data/raw
    base="https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData"
    for f in train_FD001.txt train_FD002.txt train_FD003.txt train_FD004.txt \
             test_FD001.txt test_FD002.txt test_FD003.txt test_FD004.txt \
             RUL_FD001.txt RUL_FD002.txt RUL_FD003.txt RUL_FD004.txt readme.txt; do
      curl -s -o "$f" "$base/$f"
    done

## Environment

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt

## Structure

    data/raw/                       original dataset files, untouched
    data/processed/                 cleaned / feature-engineered data
    src/turbofan_prognostics/       the actual Python package
    reports/figures/                generated plots
    notebooks/                      exploration only, not production logic
    tests/                          unit tests
    models/                         saved trained model artifacts (gitignored)
