from __future__ import annotations
import numpy as np
import pandas as pd

def make_synthetic_transactions(n=1000, seed=42) -> pd.DataFrame:
    rng=np.random.default_rng(seed); t=np.arange(n)*60
    amount=np.exp(rng.normal(3.4, 1, n)); device=rng.integers(0, 80, n).astype(str); card=rng.integers(0, 200, n).astype(str)
    risky=(amount > np.quantile(amount,.88)) | np.isin(device, ["3","7"])
    y=(rng.random(n) < np.where(risky,.22,.025)).astype(int)
    return pd.DataFrame({"TransactionID":np.arange(1,n+1),"TransactionDT":t,"TransactionAmt":amount,"ProductCD":rng.choice(list("WHC"),n),"DeviceInfo":device,"card1":card,"isFraud":y})
