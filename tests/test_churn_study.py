"""Churn study integrity and leakage-boundary tests."""
import hashlib,json
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from ml_projects.churn_study import OUT,ROOT,PROTOCOL,ChurnFeatures,choose_threshold,utility,promotion_decision
from ml_projects.data import read_dataset,split_target

def test_features_are_row_local_and_safe():
    X,_=split_target(read_dataset('churn'),'churn'); X=X.head(12)
    original=X.copy(deep=True); transform=ChurnFeatures(True).fit(X)
    full=transform.transform(X)
    pd.testing.assert_frame_equal(full.iloc[[0]],transform.transform(X.iloc[[0]]))
    pd.testing.assert_frame_equal(X,original)
    assert 'customerID' not in full and 'Churn' not in full
    zero=X.iloc[[0]].copy(); zero['tenure']=0; zero['TotalCharges']=0
    prepared=transform.transform(zero)
    assert prepared.historical_monthly_spend.isna().all()
    assert not np.isinf(prepared.select_dtypes(include='number').to_numpy()).any()
    missing=X.iloc[[0]].copy();missing['MonthlyCharges']=np.nan
    assert transform.transform(missing).monthly_charge_x_tenure.isna().all()
    assert clone(transform).engineered

def test_threshold_choice_uses_given_validation_data():
    y=np.array([0,0,1,1]);p=np.array([.1,.2,.6,.8]);threshold=choose_threshold(y,p)
    assert np.array_equal(p>=threshold,y.astype(bool))
    scenario={'value':100.,'success_rate':.1,'contact_cost':100.}
    assert choose_threshold(y,p,scenario)==1.01
    result=utility(y,p,0,{'value':500.,'success_rate':.2,'contact_cost':10.})
    assert result['assumed_net_value']==160
    assert result['unnecessary_contacts']==2 and result['missed_churners']==0

def test_promotion_rule_rejects_small_or_inconsistent_gains():
    rows=[]
    for fold in range(1,6):
        rows.extend([{'fold':fold,'procedure':'baseline','F1':.6,'log_loss':.4},{'fold':fold,'procedure':'challenger','F1':.605,'log_loss':.39}])
    assert not promotion_decision(pd.DataFrame(rows))['promote']
    for row in rows:
        if row['procedure']=='challenger':row['F1']=.63
    assert promotion_decision(pd.DataFrame(rows))['promote']
    for row in rows:
        if row['procedure']=='challenger':row['log_loss']=.41
    assert not promotion_decision(pd.DataFrame(rows))['promote']

def test_nested_partitions_and_oof_coverage():
    parts=json.loads((OUT/'partitions.json').read_text());split=json.loads((ROOT/'outputs/split.json').read_text())
    training=set(split['train_rows']);historical=set(split['test_rows']);all_eval=[]
    for part in parts:
        sets=[set(part[k]) for k in ['fit','calibration','threshold_validation','outer_evaluation']]
        assert set.union(*sets)==training
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
        assert all(not a&historical for a in sets)
        all_eval.extend(part['outer_evaluation'])
    assert len(all_eval)==len(set(all_eval))==len(training)
    predictions=pd.read_csv(OUT/'out_of_fold_predictions.csv')
    for procedure,data in predictions.groupby('procedure'):
        assert set(data.source_row)==training and not data.source_row.duplicated().any()
        assert data.probability.between(0,1).all()
        np.testing.assert_array_equal(data.prediction,(data.probability>=data.threshold).astype(int))
    assert set(predictions.procedure)=={'baseline','challenger'}

def test_recompute_outer_metrics_and_decision():
    from ml_projects.churn_study import metrics
    predictions=pd.read_csv(OUT/'out_of_fold_predictions.csv');folds=pd.read_csv(OUT/'outer_fold_metrics.csv')
    for (fold,procedure),data in predictions.groupby(['fold','procedure']):
        row=folds[(folds.fold==fold)&(folds.procedure==procedure)].iloc[0]
        result=metrics(data.actual,data.probability,row.threshold)
        for name,value in result.items():assert row[name]==pytest.approx(value)
    summary=json.loads((OUT/'summary.json').read_text()); decision=promotion_decision(folds)
    assert decision['promote']==summary['decision']['promote']
    assert decision['mean_paired_F1_gain']==pytest.approx(summary['decision']['mean_paired_F1_gain'])
    for name,digest in summary['active_model_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest

def test_business_values_and_policy_baselines():
    predictions=pd.read_csv(OUT/'out_of_fold_predictions.csv');business=pd.read_csv(OUT/'business_scenarios.csv')
    scenarios={s['name']:s for s in PROTOCOL['scenarios']}
    for _,row in business.iterrows():
        data=predictions[(predictions.fold==row.fold)&(predictions.procedure==row.procedure)]
        expected=utility(data.actual.to_numpy(),data.probability.to_numpy(),row.threshold,scenarios[row.scenario])
        for name,value in expected.items():assert row[name]==pytest.approx(value)
    assert (business[business.policy=='contact_none'].assumed_net_value==0).all()


def test_thresholds_recomputed_from_validation_only():
    val=pd.read_csv(OUT/'threshold_validation_predictions.csv')
    folds=pd.read_csv(OUT/'outer_fold_metrics.csv')
    business=pd.read_csv(OUT/'business_scenarios.csv')
    partitions={r['fold']:r for r in json.loads((OUT/'partitions.json').read_text())}
    for (fold,procedure),data in val.groupby(['fold','procedure']):
        assert set(data.source_row)==set(partitions[fold]['threshold_validation'])
        row=folds[(folds.fold==fold)&(folds.procedure==procedure)].iloc[0]
        assert choose_threshold(data.actual.to_numpy(),data.probability.to_numpy())==pytest.approx(row.threshold)
        for scenario in PROTOCOL['scenarios']:
            selected=business[(business.fold==fold)&(business.procedure==procedure)&(business.scenario==scenario['name'])&(business.policy=='validation_selected')].iloc[0]
            assert choose_threshold(data.actual.to_numpy(),data.probability.to_numpy(),scenario)==pytest.approx(selected.threshold)
