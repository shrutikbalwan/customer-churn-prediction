"""Churn improvement study. No historical-test rows enter the experiment.
Run: python -m ml_projects.churn_study
"""
import argparse,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
from html import escape
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,log_loss,brier_score_loss,roc_auc_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from threadpoolctl import threadpool_limits
from .data import ROOT,FILES,read_dataset,split_target
from .train import write_json

OUT=ROOT/'outputs'/'churn_study'
SERVICES=['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
PROTOCOL={'seed':20261127,'outer_folds':5,'inner_folds':3,'selection_metric':'neg_log_loss','classification_threshold_objective':'validation F1','threshold_grid':[round(float(v),2) for v in np.linspace(.05,.95,91)],'fit_fraction_of_outer_training':.6,'calibration_fraction_of_outer_training':.2,'threshold_fraction_of_outer_training':.2,'logistic_C':[.1,1.],'boosting_leaf_nodes':[15,31],'boosting_iterations':150,'promotion':{'min_mean_F1_gain':.01,'min_fold_wins':4,'max_mean_log_loss_increase':0.0},'scenarios':[{'name':'low_contact_cost','value':500.,'success_rate':.2,'contact_cost':10.},{'name':'moderate_contact_cost','value':500.,'success_rate':.2,'contact_cost':30.},{'name':'high_contact_cost','value':500.,'success_rate':.2,'contact_cost':60.}],'scope':'Original training rows only. The historical test partition is excluded. Earlier development has used this dataset; nested CV is internal validation, not external evidence.'}

class ChurnFeatures(TransformerMixin,BaseEstimator):
    """Row-local features only; no target or cohort statistics are learned."""
    def __init__(self,engineered=False): self.engineered=engineered
    def fit(self,X,y=None):
        self.feature_names_in_=np.array(X.columns,dtype=object);return self
    def transform(self,X):
        frame=X.copy().drop(columns=['customerID','Churn'],errors='ignore').replace(r'^\s*$',np.nan,regex=True)
        for col in ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen']: frame[col]=pd.to_numeric(frame[col],errors='coerce')
        if self.engineered:
            tenure=frame.tenure; charge=frame.MonthlyCharges
            frame['tenure_band']=pd.cut(tenure,[-np.inf,0,6,12,24,48,np.inf],labels=['zero','1-6','7-12','13-24','25-48','49+']).astype(object)
            # min_count preserves a completely missing service profile as missing.
            flags=frame[SERVICES].apply(lambda col:col.map(lambda v:np.nan if pd.isna(v) else float(v=='Yes')))
            frame['service_count']=flags.sum(axis=1,min_count=1)
            frame['support_count']=flags[['OnlineSecurity','TechSupport']].sum(axis=1,min_count=1)
            frame['historical_monthly_spend']=frame.TotalCharges/tenure.where(tenure>0)
            frame['current_vs_historical_spend']=charge-frame.historical_monthly_spend
            frame['charge_per_service']=charge/frame.service_count.where(frame.service_count>0)
            frame['monthly_charge_x_tenure']=charge*tenure
            frame['contract_x_internet']=frame.Contract.fillna('Missing').astype(str)+' / '+frame.InternetService.fillna('Missing').astype(str)
            frame['autopay']=frame.PaymentMethod.map(lambda v:np.nan if pd.isna(v) else float('automatic' in str(v).lower()))
        return frame.replace([np.inf,-np.inf],np.nan)

def estimator(seed):
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])
    categorical=Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    preprocess=ColumnTransformer([('numeric',numeric,make_column_selector(dtype_include=np.number)),('categorical',categorical,make_column_selector(dtype_exclude=np.number))])
    return Pipeline([('features',ChurnFeatures()),('preprocess',preprocess),('model',LogisticRegression(max_iter=3000,random_state=seed))])

def parameters(challenger,seed):
    grid=[{'features__engineered':[False,True] if challenger else [False], 'model':[LogisticRegression(max_iter=3000,random_state=seed)],'model__C':PROTOCOL['logistic_C']}]
    if challenger: grid.append({'features__engineered':[False,True],'model':[HistGradientBoostingClassifier(random_state=seed,max_iter=PROTOCOL['boosting_iterations'])],'model__max_leaf_nodes':PROTOCOL['boosting_leaf_nodes'],'model__l2_regularization':[1.]})
    return grid

def metrics(y,p,threshold):
    pred=p>=threshold;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'F1':float(f1_score(y,pred,zero_division=0)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'Brier':float(brier_score_loss(y,p)),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def utility(y,p,threshold,scenario):
    selected=p>=threshold;tp=int(np.sum(selected & (np.asarray(y)==1)));contacts=int(selected.sum())
    gain=tp*scenario['value']*scenario['success_rate']-contacts*scenario['contact_cost']
    return {'contacts':contacts,'true_churners_contacted':tp,'unnecessary_contacts':contacts-tp,'missed_churners':int(np.sum((~selected)&(np.asarray(y)==1))),'assumed_net_value':float(gain),'assumed_net_value_per_1000':float(gain/len(y)*1000)}

def choose_threshold(y,p,scenario=None):
    grid=[*PROTOCOL['threshold_grid'],1.01] if scenario else PROTOCOL['threshold_grid']
    score=lambda t:utility(y,p,t,scenario)['assumed_net_value'] if scenario else f1_score(y,p>=t,zero_division=0)
    # A higher threshold wins exact ties to contact fewer customers.
    return float(max(grid,key=lambda t:(score(t),t)))

def promotion_decision(folds):
    wide=folds.pivot(index='fold',columns='procedure',values='F1');gains=wide.challenger-wide.baseline
    losses=folds.pivot(index='fold',columns='procedure',values='log_loss')
    loss_delta=float((losses.challenger-losses.baseline).mean());rule=PROTOCOL['promotion']
    promote=bool(gains.mean()>=rule['min_mean_F1_gain'] and (gains>0).sum()>=rule['min_fold_wins'] and loss_delta<=rule['max_mean_log_loss_increase'])
    return {'promote':promote,'mean_paired_F1_gain':float(gains.mean()),'fold_wins':int((gains>0).sum()),'mean_log_loss_change':loss_delta,'rule':rule,'interpretation':'A conservative engineering gate, not a statistical significance test. Outer folds overlap in training data.'}

def error_groups(X,y,predictions):
    rows=[]
    for procedure,data in predictions.groupby('procedure'):
        data=data.set_index('source_row'); ids=data.index; actual=y.loc[ids].to_numpy();pred=data.prediction.to_numpy()
        groups={'Contract':X.loc[ids,'Contract'].astype(str),'InternetService':X.loc[ids,'InternetService'].astype(str),'tenure_band':pd.cut(X.loc[ids,'tenure'],[-np.inf,6,12,24,48,np.inf],labels=['0-6','7-12','13-24','25-48','49+']).astype(str),'MonthlyCharges_band':pd.cut(X.loc[ids,'MonthlyCharges'],[-np.inf,35,70,100,np.inf],labels=['under 35','35-70','70-100','100+']).astype(str)}
        for attribute,values in groups.items():
            for value in sorted(values.unique()):
                mask=(values==value).to_numpy();tn,fp,fn,tp=confusion_matrix(actual[mask],pred[mask],labels=[0,1]).ravel()
                rows.append({'procedure':procedure,'attribute':attribute,'group':value,'rows':int(mask.sum()),'actual_churners':int(tp+fn),'false_positives':int(fp),'false_negatives':int(fn),'false_positive_rate':float(fp/(fp+tn)) if fp+tn else None,'miss_rate':float(fn/(fn+tp)) if fn+tp else None,'F1':float(f1_score(actual[mask],pred[mask],zero_division=0)),'small_sample':bool(mask.sum()<100)})
    return pd.DataFrame(rows)

def run():
    OUT.mkdir(parents=True,exist_ok=True);write_json(OUT/'protocol.json',PROTOCOL)
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'outputs/advanced/model.joblib',ROOT/'outputs/advanced/summary.json',ROOT/'outputs/model.joblib']}
    X,y=split_target(read_dataset('churn'),'churn'); X=X.drop(columns=['customerID'])
    split=json.loads((ROOT/'outputs/split.json').read_text()); ids=np.array(split['train_rows']);excluded=set(split['test_rows'])
    assert not set(ids)&excluded
    outer=StratifiedKFold(PROTOCOL['outer_folds'],shuffle=True,random_state=PROTOCOL['seed'])
    folds=[];predictions=[];business=[];partitions=[];searches=[];validation_predictions=[]
    started=time.time()
    for fold,(train_pos,test_pos) in enumerate(outer.split(X.loc[ids],y.loc[ids]),1):
        train_ids,eval_ids=ids[train_pos],ids[test_pos]; seed=PROTOCOL['seed']+fold
        fit,held=train_test_split(train_ids,test_size=.4,stratify=y.loc[train_ids],random_state=seed)
        cal,val=train_test_split(held,test_size=.5,stratify=y.loc[held],random_state=seed)
        partitions.append({'fold':fold,'fit':fit.tolist(),'calibration':cal.tolist(),'threshold_validation':val.tolist(),'outer_evaluation':eval_ids.tolist()})
        for label,challenger in [('baseline',False),('challenger',True)]:
            print(f'Outer fold {fold}/5: {label}',flush=True)
            inner=StratifiedKFold(PROTOCOL['inner_folds'],shuffle=True,random_state=seed)
            search=GridSearchCV(estimator(seed),parameters(challenger,seed),scoring=PROTOCOL['selection_metric'],cv=inner,n_jobs=1,error_score='raise')
            search.fit(X.loc[fit],y.loc[fit]);base=search.best_estimator_
            pd.DataFrame(search.cv_results_).to_csv(OUT/f'fold_{fold}_{label}_inner_search.csv',index=False)
            model=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid',n_jobs=1);model.fit(X.loc[cal],y.loc[cal])
            vp=model.predict_proba(X.loc[val])[:,1];threshold=choose_threshold(y.loc[val].to_numpy(),vp)
            for row,actual,probability in zip(val,y.loc[val],vp):
                validation_predictions.append({'fold':fold,'procedure':label,'source_row':int(row),'actual':int(actual),'probability':float(probability)})
            p=model.predict_proba(X.loc[eval_ids])[:,1];truth=y.loc[eval_ids].to_numpy()
            name=type(base.named_steps['model']).__name__; engineered=base.named_steps['features'].engineered
            folds.append({'fold':fold,'procedure':label,'model':name,'engineered':bool(engineered),'threshold':threshold,**metrics(truth,p,threshold)})
            searches.append({'fold':fold,'procedure':label,'inner_cv_score':float(search.best_score_),'parameters':{k:str(v) if k=='model' else v for k,v in search.best_params_.items()}})
            for row,actual,probability in zip(eval_ids,truth,p):predictions.append({'fold':fold,'procedure':label,'source_row':int(row),'actual':int(actual),'probability':float(probability),'prediction':int(probability>=threshold),'threshold':threshold})
            for scenario in PROTOCOL['scenarios']:
                bt=choose_threshold(y.loc[val].to_numpy(),vp,scenario)
                for policy,t in [('validation_selected',bt),('contact_all',0.),('contact_none',1.01)]:
                    business.append({'fold':fold,'procedure':label,'scenario':scenario['name'],'policy':policy,'threshold':t,**utility(truth,p,t,scenario)})
    fold_df=pd.DataFrame(folds);pred_df=pd.DataFrame(predictions);business_df=pd.DataFrame(business)
    fold_df.to_csv(OUT/'outer_fold_metrics.csv',index=False);pred_df.to_csv(OUT/'out_of_fold_predictions.csv',index=False)
    pd.DataFrame(validation_predictions).to_csv(OUT/'threshold_validation_predictions.csv',index=False)
    business_df.to_csv(OUT/'business_scenarios.csv',index=False);write_json(OUT/'partitions.json',partitions);write_json(OUT/'inner_selections.json',searches)
    errors=error_groups(X,y,pred_df);errors.to_csv(OUT/'error_groups.csv',index=False)
    # Save error rows for local inspection without customer identifiers.
    pred_df[pred_df.actual!=pred_df.prediction].to_csv(OUT/'out_of_fold_errors.csv',index=False)
    decision=promotion_decision(fold_df)
    means=fold_df.groupby('procedure')[['F1','precision','recall','ROC_AUC','log_loss','Brier']].mean().reset_index().to_dict('records')
    after={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in before}; assert before==after
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'training_rows':len(ids),'excluded_historical_test_rows':len(excluded),'source_sha256':hashlib.sha256((ROOT/'datasets'/FILES['churn']).read_bytes()).hexdigest(),'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol':PROTOCOL,'mean_outer_metrics':means,'decision':decision,'active_models_unchanged':before==after,'active_model_hashes':after,'elapsed_seconds':time.time()-started,'scope':'Nested evaluation of two learning procedures, not a direct comparison with an already-fitted model. No historical-test rows or metrics used. Prior development history limits claims of novelty.'}
    write_json(OUT/'summary.json',report)
    write_report(report,errors,business_df,fold_df)
    print(json.dumps({'mean_outer_metrics':means,'decision':decision},indent=2),flush=True)
    return report

def write_report(summary,errors,business,folds):
    def table(rows):return pd.DataFrame(rows).to_html(index=False,escape=True,float_format=lambda v:f'{v:.4f}')
    d=summary['decision'];status='Challenger meets the predefined gate; candidate promotion requires a separately verified final fit.' if d['promote'] else 'Keep the current application model. The challenger did not meet the predefined promotion gate.'
    worst=errors[(errors.procedure=='baseline') & (~errors.small_sample)].sort_values('false_negatives',ascending=False).head(8)
    worst_fp=errors[(errors.procedure=='baseline') & (~errors.small_sample)].sort_values('false_positives',ascending=False).head(8)
    policies=business.groupby(['procedure','scenario','policy'])[['assumed_net_value_per_1000','contacts','missed_churners','unnecessary_contacts']].mean().reset_index()
    html='<h1>Churn improvement study</h1><p>'+escape(status)+'</p><h2>Paired nested evaluation</h2>'+table(summary['mean_outer_metrics'])+'<p>Mean metrics across five outer folds. Every outer evaluation row was excluded from fitting, calibration and threshold selection for that fold. The original 1,409 historical-test rows were excluded throughout.</p><h2>Decision</h2><pre>'+escape(json.dumps(d,indent=2))+'</pre><h2>Fold-by-fold evidence</h2>'+table(folds)+'<h2>Where missed churn occurs</h2>'+table(worst)+'<h2>Where false alarms occur</h2>'+table(worst_fp)+'<p>Groups overlap and cannot be summed. These are out-of-fold errors from original training rows, examined after the experiment. Future feature hypotheses need a new protocol; repeated tuning on these results would overfit this evaluation.</p><h2>Retention scenarios</h2><p>Hypothetical retained-customer value: 500 currency units; assumed intervention success: 20%; per-contact costs: 10, 30 or 60. Expected value = true churners contacted × 500 × 0.20 − all contacts × cost. These are not measured causal effects, realized profit or uplift estimates. Contact costs apply to everyone selected; missed churners represent forgone assumed opportunities.</p>'+table(policies)+'<p>Thresholds were selected on inner validation rows and evaluated on outer rows. A threshold above 1 is the explicit contact-nobody policy. Compare against contact-all and contact-none before making a decision.</p><h2>Features tested</h2><p>Tenure bands, service/support counts, historical monthly spending, spending changes, charge per service, spending-tenure interaction, contract/internet combinations and automatic-payment indicators. All are row-local; learned imputation, scaling and encoding remain inside each fold. No customer identifier or target is a feature. Historical charges are assumed available at scoring time; the dataset does not prove temporal availability.</p><h2>Limits and reproducibility</h2><p>This is internal validation of previously studied data, not external or causal validation. The baseline is a freshly trained calibrated logistic procedure on identical partitions, not the saved incumbent. Fold estimates are dependent because training rows overlap. The promotion gate is an engineering rule, not a significance test. Active app artifacts were left unchanged.</p><p>Run <code>python -m ml_projects.churn_study</code> from this project folder. Protocol, row partitions, inner searches, out-of-fold predictions, errors and business tables are in <code>outputs/churn_study/</code>.</p><p>Method: <a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html">scikit-learn nested cross-validation</a>.</p>'
    css='body{font:16px/1.6 Segoe UI,sans-serif;color:#163044;max-width:1280px;margin:40px auto;padding:0 25px;background:#f5f8fa}h1{font-size:36px}h2{margin-top:32px}table{border-collapse:collapse;background:white;font-size:14px;display:block;overflow:auto}th,td{padding:10px;border-bottom:1px solid #dbe5ea;text-align:left}th{background:#e6eff3}pre{background:#eaf0f4;padding:20px;overflow:auto}a{color:#146b62}'
    (ROOT/'CHURN_IMPROVEMENT_REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Churn improvement study</title><style>'+css+'</style><body>'+html+'</body></html>',encoding='utf-8')

if __name__=='__main__':
    from pathlib import Path
    with threadpool_limits(limits=1): run()
