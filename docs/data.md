# Data guide: IEEE-CIS

1. Obtain the Kaggle **IEEE-CIS Fraud Detection** competition data with your own account and accept its rules.
2. Place `train_transaction.csv` and, if used, `train_identity.csv` outside version control, e.g. `data/ieee-cis/`.
3. Load with `fraudml.data.load_ieee_cis(...)`; the merge key is `TransactionID`.
4. Use `TransactionDT` for chronological splitting. Treat it as a relative time measure and document the cutoff indices/timestamps used. `feature_columns()` drops `TransactionDT` from model features by default: its absolute values differ systematically between periods, so using it as a raw feature leaks temporal structure.

Limitations: competition data is anonymized, labels are historical, missingness may be informative but should be handled consistently, and class imbalance means accuracy is misleading. Never use `isFraud` or post-outcome information as a feature. Do not fit encoders/imputers on validation or test data. Do not report Kaggle leaderboard scores as unbiased test estimates.
