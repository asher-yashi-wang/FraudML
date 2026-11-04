# Usage

## Python API

```python
from fraudml.data import chronological_split, feature_columns
from fraudml.models import TabularFraudModel
import pandas as pd
df = pd.read_csv("data/demo.csv")
s = chronological_split(df); cols = feature_columns(df)
model = TabularFraudModel("logistic").fit(s.train[cols], s.train.isFraud)
probabilities = model.predict_proba(s.test[cols])
model.save("artifacts/model.joblib")
```

## Preprocessing notes

`feature_columns()` excludes `TransactionID`, `isFraud`, and the split time column `TransactionDT` by default. `build_preprocessor()` one-hot encodes categorical columns with at most 32 distinct values and frequency-encodes higher-cardinality columns (e.g. `card1`, `addr1`) so the feature matrix stays sparse and tractable; both encoders are fitted on training data only and unseen values map to 0 frequency.

## GNN installation and protocol

Install PyTorch for the installed NVIDIA driver/CUDA version first, then a matching PyG release; their official install matrix is authoritative. Install this package with `pip install -e '.[gnn]'`. `build_causal_bipartite_edges` returns edges whose `transaction_index` is the positional row of the time-sorted frame (it resets the index internally); build node features on the same `sort_values(...).reset_index(drop=True)` frame so indices align. It remains a building block: create separate train/validation/test graph snapshots and ensure no future graph information participates in messages. Record CUDA, PyTorch, PyG, driver versions, GPU RAM, batch size, sampling fanout, seed, and OOM adjustments.
