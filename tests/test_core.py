import numpy as np
import pandas as pd
from fraudml.data import chronological_split, feature_columns
from fraudml.features import FrequencyEncoder, add_past_entity_stats
from fraudml.metrics import top_k_metrics
from fraudml.models import TabularFraudModel
from fraudml.synthetic import make_synthetic_transactions
from fraudml.policy import RiskPolicy
from fraudml.workflow import review_queue, triage

def test_time_split_has_no_future_train_rows():
    df=make_synthetic_transactions(100)
    s=chronological_split(df)
    assert s.train.TransactionDT.max() < s.valid.TransactionDT.min() < s.test.TransactionDT.min()

def test_feature_columns_excludes_time_and_id_and_target():
    df=make_synthetic_transactions(10)
    cols=feature_columns(df)
    assert "TransactionDT" not in cols and "TransactionID" not in cols and "isFraud" not in cols
    assert "TransactionAmt" in cols

def test_top_k_metrics():
    r=top_k_metrics([1,0,1,0],[.9,.8,.7,.1],2)
    assert r == {"k":2,"precision_at_k":.5,"recall_at_k":.5,"hits_at_k":1}

def test_logistic_fit_and_probability():
    df=make_synthetic_transactions(400); s=chronological_split(df); cols=feature_columns(df)
    m=TabularFraudModel().fit(s.train[cols],s.train.isFraud)
    p=m.predict_proba(s.test[cols])
    assert len(p)==len(s.test) and np.all((p>=0)&(p<=1))

def test_frequency_encoder_unseen_and_missing_map_to_zero():
    enc=FrequencyEncoder().fit(pd.DataFrame({"c":["a","a","b","c"]}))
    out=enc.transform(pd.DataFrame({"c":["a","zzz",None]}))
    assert out.iloc[0,0]==0.5 and out.iloc[1,0]==0.0 and out.iloc[2,0]==0.0

def test_past_entity_stats_match_naive_reference():
    df=make_synthetic_transactions(120)
    out=add_past_entity_stats(df, ["card1", "DeviceInfo"])
    ref=df.sort_values("TransactionDT",kind="stable").reset_index(drop=True)
    for col in ["card1","DeviceInfo"]:
        g=ref.groupby(col, dropna=False, sort=False)["TransactionAmt"]
        naive_count=g.cumcount().to_numpy(float)
        naive_mean=g.transform(lambda s: s.shift().expanding().mean()).to_numpy()
        got_count=out[f"{col}__past_count"].to_numpy(float)
        got_mean=out[f"{col}__past_amount_mean"].to_numpy()
        assert np.array_equal(got_count,naive_count)
        assert np.allclose(got_mean,naive_mean,equal_nan=True)

def test_policy_triage_keeps_confirmation_for_human_review():
    df=make_synthetic_transactions(3); df["risk_score"]=[.1,.5,.95]
    result=triage(df, RiskPolicy(reviewer="tester", review_threshold=.4, challenge_threshold=.8))
    assert result.decision.tolist()==["APPROVE","REVIEW","CHALLENGE"]
    assert "confirmed_fraud" not in result.columns
    assert len(review_queue(result))==1

def test_triage_vectorized_blocked_values_and_case_ids():
    df=make_synthetic_transactions(4); df["risk_score"]=[.1,.5,.95,.2]
    policy=RiskPolicy(reviewer="tester", review_threshold=.4, challenge_threshold=.8,
                      blocked_values={"DeviceInfo":["3"]})
    df["DeviceInfo"]=["3","9","3","9"]
    result=triage(df, policy)
    assert result.decision.tolist()==["CHALLENGE","REVIEW","CHALLENGE","APPROVE"]
    assert all(result.review_case_id.str.startswith("case-"))
    assert all(result.review_case_id.str.len()==17)
    assert "policy blocked value: DeviceInfo" in result.decision_reason.iloc[0]
    assert "score below review threshold" in result.decision_reason.iloc[3]
