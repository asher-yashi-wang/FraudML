# Reproducing FraudML results

## Inspect recorded metrics after cloning

The four JSON files in artifacts are committed with the source. Run the following command with standard Python.

```bash
python scripts/verify_results.py
```

It checks each report and prints its SHA-256 hash. This validates the recorded counts and metric arithmetic. Reproducing the numerical training outcomes also requires the original data and a compatible environment.

## Recreate the environment and experiments

Download IEEE-CIS through your own Kaggle account under its data terms. Put the original labeled train_transaction.csv into data/ieee-cis. The recorded runs used this file without the identity join.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tested.txt
python -m pip install -e .
python -m pytest -q
python scripts/run_experiment.py --transaction data/ieee-cis/train_transaction.csv --categorical-codes card1 addr1 --k 100 500 --out artifacts/reproduced_validation_base.json
python scripts/run_experiment.py --transaction data/ieee-cis/train_transaction.csv --categorical-codes card1 addr1 --entity-cols card1 addr1 --models lightgbm --k 100 500 --out artifacts/reproduced_validation_history.json
```

On macOS, LightGBM needs the OpenMP runtime. Install libomp with Homebrew if import fails because libomp.dylib is missing. Windows users activate .venv with its Scripts/Activate.ps1.

The requirements snapshot records package versions observed on the author's Python 3.9 macOS environment on 5 November 2026. The original experiment reports did not record a dependency lock or hardware fingerprint, so this snapshot cannot certify the historical execution environment. Platform, thread scheduling and library versions can affect exact results.

Once a protocol is frozen, its planned final test evaluations add --split test and distinct output paths. The existing final reports are retained as the primary results. A retraining run for verification should be identified as replication rather than another model-selection opportunity.

## Reproduce service integration

```bash
python scripts/smoke_service.py
```

This command requires JDK 11+ on PATH. Alternatively provide --java-home followed by the JDK directory. It trains a small synthetic model, compiles the Java client, starts a temporary local HTTP server, calls it and checks score equality. It does not require IEEE-CIS files.

The integration was executed successfully on 5 November 2026 with a local JDK 17 compiling for Java 11 and the Python scoring stack. This is a local integration check. Production hosting, bank deployment and serving load benchmarks remain open work.
