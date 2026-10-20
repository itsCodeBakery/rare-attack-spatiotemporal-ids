"""E7 G3 source-hash-verified, day-disjoint, flow-only baselines.

Never uses source-row neighbors as time, raw endpoints as features, or frozen UNSW tests.
All train/validation/test folds come directly from the preregistered G3 manifest.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score,average_precision_score,confusion_matrix,
                             balanced_accuracy_score,precision_recall_fscore_support,
                             accuracy_score)

BANNED={'flowid','sourceip','sourceport','destinationip','destinationport',
        'timestamp','label','sourcefilename','sourceday','day','rowindex'}
SEEDS=(42,43,44,45,46)
PARAMS={'objective':'binary','metric':'binary_logloss','learning_rate':0.05,
        'num_leaves':31,'min_data_in_leaf':120,'feature_fraction':1.0,
        'bagging_fraction':0.85,'bagging_freq':1,'max_bin':255,
        'num_threads':4,'verbosity':-1,'deterministic':True,'force_col_wise':True}
ROUNDS=180
THRESHOLDS=np.linspace(0.0,1.0,201,dtype=np.float64)
FP_COST=1
FN_COST=5

def normalize(s):
    return ''.join(c.lower() for c in str(s) if c.isalnum())

def fingerprint(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def check_protocol(p):
    assert p['protocol_id']=='E7-G3-OPENWORLD-DAYHOLDOUT-001'
    assert p['status']=='PREREGISTERED_NO_G3_MODELS_TRAINED'
    assert len(p['source_files'])==8 and len(p['folds'])==4
    assert p['derived_nonempty_rows_total']==2830743
    days=set(p['day_class_support'])
    assert days=={'Monday','Tuesday','Wednesday','Thursday','Friday'}
    for fold in p['folds']:
        assert len(set(fold['train_days']+[fold['validation_day'],fold['test_day']]))==5
        assert set(fold['train_days']+[fold['validation_day'],fold['test_day']])==days
        assert not set(fold['unseen_test_classes']).intersection(fold['known_train_classes'])
        assert fold['unseen_test_classes']
    return True

def select_features(columns):
    normalized=[normalize(c) for c in columns]
    if len(set(normalized))!=len(normalized):
        raise ValueError('Duplicate normalized source column names; do not choose silently')
    mapping=dict(zip(normalized,columns))
    required={'sourceip','destinationip','sourceport','destinationport',
              'flowid','timestamp','label','protocol'}
    assert required.issubset(mapping),'Required source schema missing'
    features=[c for c in columns if normalize(c) not in BANNED]
    assert len(features)>=65,'Unexpected source feature count'
    assert 'protocol' in [normalize(c) for c in features]
    assert all(normalize(f) not in BANNED for f in features)
    return features

def read_source(path,spec,expected_features=None,chunk_size=25000):
    path=Path(path)
    if (path.name!=spec['source_file'] or not path.is_file()
            or path.suffix.lower()!='.csv' or 'unsw' in path.name.lower()):
        raise ValueError('Only manifest-named original CIC source allowed')
    assert spec['encoding'] in ('utf-8-sig','cp1252')
    assert fingerprint(path)==spec['source_file_sha256'],'Original source SHA256 mismatch'
    arrays=[];targets=[];labels=[];original=0;removed=0;totals=Counter()
    cols_seen=None
    for chunk in pd.read_csv(path,encoding=spec['encoding'],dtype='string',
                             chunksize=chunk_size,low_memory=False,on_bad_lines='error'):
        if cols_seen is None:
            cols_seen=list(chunk.columns)
            feature_names=select_features(cols_seen)
            if expected_features is not None:
                assert feature_names==expected_features,'Schema or feature order mismatch'
            label_col=next(c for c in cols_seen if normalize(c)=='label')
        else:
            assert list(chunk.columns)==cols_seen,'Original source headers changed'
        original+=len(chunk)
        raw_label=chunk[label_col].astype('string').str.strip()
        label_missing=raw_label.isna()|raw_label.fillna('').eq('')
        if bool(label_missing.any()):
            candidates=chunk.loc[label_missing].fillna('')
            truly_blank=candidates.apply(lambda col:col.str.strip().eq('')).all(axis=1)
            if not bool(truly_blank.all()):
                raise ValueError('Nonempty original flow without label; halt')
        removed+=int(label_missing.sum())
        if bool(label_missing.all()):continue
        active=chunk.loc[~label_missing]
        y_text=raw_label.loc[~label_missing]
        totals.update({str(k):int(v) for k,v in y_text.value_counts().items()})
        x=active[feature_names].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64,na_value=np.nan)
        x[~np.isfinite(x)]=np.nan
        x=np.where(np.abs(x)>np.finfo(np.float32).max,np.nan,x).astype(np.float32)
        arrays.append(x)
        targets.append(y_text.ne('BENIGN').to_numpy(dtype=np.uint8))
        labels.append(y_text.to_numpy(dtype=str))
    assert cols_seen is not None and arrays,'Empty or all-invalid source'
    assert original==spec['original_rows']
    assert removed==spec['fully_empty_rows_verified_by_g2']
    assert original-removed==spec['derived_nonempty_rows']
    assert dict(sorted(totals.items()))==dict(sorted(spec['original_labels'].items())),(
        'Original source label counts drifted',path.name)
    X=np.concatenate(arrays,axis=0);Y=np.concatenate(targets,axis=0)
    L=np.concatenate(labels,axis=0)
    assert len(Y)==spec['derived_nonempty_rows']
    assert np.all(np.isfinite(X)|np.isnan(X))
    return X,Y,L,feature_names,{'original_rows':original,'blank_rows_omitted':removed}

def build_all(dataset_root,protocol,chunk_size=25000):
    check_protocol(protocol)
    root=Path(dataset_root)
    assert root.exists()
    csv_files=sorted(root.rglob('*.csv'))
    allowed={s['source_file'] for s in protocol['source_files']}
    assert len(csv_files)==8 and {f.name for f in csv_files}==allowed
    assert len({str(f.parent) for f in csv_files})==1
    assert csv_files[0].parent.name=='TrafficLabelling ','Unexpected directory (trailing space expected)'
    by_day={};features=None;provenance=[]
    lookup={p.name:p for p in csv_files}
    for source in protocol['source_files']:
        p=lookup[source['source_file']]
        X,Y,L,fn,stat=read_source(p,source,features,chunk_size)
        if features is None:features=fn
        day=source['source_day']
        by_day.setdefault(day,{'X':[],'y':[],'labels':[]})
        for key,v in [('X',X),('y',Y),('labels',L)]:by_day[day][key].append(v)
        provenance.append({'source':p.name,**stat,'sha256':source['source_file_sha256']})
        print('SOURCE CHECKED',p.name,'flows',len(Y),'attack',int(Y.sum()),
              'blank skipped',stat['blank_rows_omitted'],flush=True)
    for day,v in by_day.items():
        for key in ('X','y','labels'):v[key]=np.concatenate(v[key],axis=0)
        assert len(v['y'])==protocol['day_class_support'][day]['rows']
        assert int((v['y']==0).sum())==protocol['day_class_support'][day]['benign_rows']
    assert sum(len(v['y']) for v in by_day.values())==protocol['derived_nonempty_rows_total']
    return by_day,features,provenance

def get_fold(by_day,fold):
    tr=fold['train_days'];vd=fold['validation_day'];td=fold['test_day']
    assert len(set(tr+[vd,td]))==5
    tx=np.concatenate([by_day[d]['X'] for d in tr])
    ty=np.concatenate([by_day[d]['y'] for d in tr])
    val=by_day[vd];test=by_day[td]
    assert np.unique(ty).tolist()==[0,1]
    assert np.unique(val['y']).tolist()==[0,1]
    assert np.unique(test['y']).tolist()==[0,1]
    known=set(np.concatenate([by_day[d]['labels'] for d in tr]).tolist())
    unseen=set(test['labels'].tolist())-{'BENIGN'}
    assert unseen==set(fold['unseen_test_classes']) and not unseen.intersection(known)
    return tx,ty,val['X'],val['y'],test['X'],test['y'],test['labels']

def train_seed(X,y,features,seed):
    assert seed in SEEDS
    params={**PARAMS,'seed':seed,'bagging_seed':seed,'feature_fraction_seed':seed}
    data=lgb.Dataset(X,label=y,feature_name=features,free_raw_data=True)
    return lgb.train(params,data,num_boost_round=ROUNDS)

def choose_threshold(y,proba):
    assert len(y)==len(proba) and len(y)>0
    assert np.all(np.isfinite(proba)) and np.all((proba>=0)&(proba<=1))
    costs=np.array([FP_COST*int(np.sum((proba>=t)&(y==0)))+
                    FN_COST*int(np.sum((proba<t)&(y==1))) for t in THRESHOLDS],
                   dtype=np.int64)
    winners=np.flatnonzero(costs==costs.min())
    idx=int(winners[-1])
    return {'threshold':float(THRESHOLDS[idx]),'validation_cost':int(costs[idx]),
            'validation_rows':int(len(y)),'fp_cost':FP_COST,'fn_cost':FN_COST,
            'grid_points':int(len(THRESHOLDS)),'tie_break':'largest_threshold'}

def evaluate(y,prob,threshold,raw_labels=None):
    y=np.asarray(y,dtype=np.uint8);prob=np.asarray(prob,dtype=np.float64)
    assert len(y)==len(prob) and set(np.unique(y))=={0,1}
    assert np.all(np.isfinite(prob)) and np.all((prob>=0)&(prob<=1))
    output=(prob>=threshold).astype(np.uint8)
    tn,fp,fn,tp=confusion_matrix(y,output,labels=[0,1]).ravel()
    precision,recall,f1,support=precision_recall_fscore_support(
        y,output,labels=[0,1],zero_division=0)
    result={
      'threshold':float(threshold),'accuracy':float(accuracy_score(y,output)),
      'balanced_accuracy':float(balanced_accuracy_score(y,output)),
      'auroc':float(roc_auc_score(y,prob)),
      'average_precision':float(average_precision_score(y,prob)),
      'macro_f1':float(np.mean(f1)),'binary_f1':float(f1[1]),
      'precision_attack':float(precision[1]),'recall_attack':float(recall[1]),
      'fpr':float(fp/(fp+tn)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
      'rows':int(len(y)),'attack_support':int(y.sum()),
      'benign_support':int((y==0).sum()),'cost_5fn_1fp':int(5*fn+fp)}
    if raw_labels is not None:
        per={}
        for label in sorted(set(raw_labels)-{'BENIGN'}):
            mask=raw_labels==label
            per[str(label)]={'support':int(mask.sum()),'recall':float(output[mask].mean()),
                 'low_support':bool(mask.sum()<100)}
        result['unseen_family_recall']=per
    return result
