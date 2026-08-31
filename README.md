# Airworthy

Physics-informed hybrid Remaining Useful Life (RUL) model for turbofan engines
(NASA C-MAPSS), wrapped in a full MLOps loop (MLflow, Evidently drift
detection + retraining, CI/CD, Prometheus/Grafana), plus a LangGraph agent
that cross-checks live FAA Airworthiness Directives and drafts
compliance-aware maintenance recommendations, with n8n routing approvals.

Status: setup in progress. Build log / decisions will be added here as we go.

## Dataset

NASA Turbofan Engine Degradation Simulation (C-MAPSS), 4 sub-datasets
(FD001-FD004), simulated multivariate sensor time series for a fleet of
engines run to failure under varying operating conditions and fault modes.
Not committed to git (see data/raw/, gitignored) — re-fetch instructions
will go here.

## Environment

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt

## Structure

    data/raw/          original dataset files, untouched
    data/processed/     cleaned / feature-engineered data we generate
    src/airworthy/      the actual Python package
    notebooks/           exploration only, not production logic
    tests/                unit tests
    models/               saved trained model artifacts (gitignored)
