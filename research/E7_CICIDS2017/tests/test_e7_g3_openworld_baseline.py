"""Synthetic E7 G3 preflight: no real traffic, no Kaggle and no UNSW test."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/e7_g3_openworld_baseline.py'
spec=importlib.util.spec_from_file_location('g3_training',SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def synthetic_source(tmp_path):
    p=tmp_path/'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    headers=['Flow ID',' Source IP',' Source Port',' Destination IP',
             ' Destination Port',' Protocol',' Timestamp']
    headers.extend(' Numeric Feature '+str(i) for i in range(70))
    headers+=[' Label']
    rows=[]
    for i,label in enumerate(['BENIGN','Web Attack – XSS',None]):
        row=['flow'+str(i),'10.0.0.'+str(i+1),'100','192.168.1.2','80',
             '6','6/7/2017 8:59']
        row+=[str(j+i) for j in range(70)]
        row+=[label]
        if label is None:row=[None]*len(headers)
        rows.append(row)
    pd.DataFrame(rows,columns=headers).to_csv(p,index=False,encoding='cp1252')
    spec={'source_file':p.name,'source_file_sha256':m.fingerprint(p),
       'original_rows':3,'derived_nonempty_rows':2,
       'fully_empty_rows_verified_by_g2':1,'encoding':'cp1252',
       'original_labels':{'BENIGN':1,'Web Attack – XSS':1}}
    return p,spec

def test_derived_only_blank_row_exclusion_and_source_immutable(tmp_path):
    file,source=synthetic_source(tmp_path)
    before=file.read_bytes()
    X,Y,L,features,info=m.read_source(file,source,chunk_size=1)
    assert file.read_bytes()==before
    assert X.shape==(2,71)
    assert Y.tolist()==[0,1]
    assert L.tolist()==['BENIGN','Web Attack – XSS']
    assert info['blank_rows_omitted']==1
    assert all(m.normalize(f) not in m.BANNED for f in features)
    assert ' Flow ID' not in features and ' Source IP' not in features
    assert m.select_features(pd.read_csv(file,nrows=1).columns)==features

def test_nonempty_missing_label_halts(tmp_path):
    file,source=synthetic_source(tmp_path)
    df=pd.read_csv(file,dtype='string',encoding='cp1252')
    df.loc[2,' Numeric Feature 3']='345'
    df.to_csv(file,index=False,encoding='cp1252')
    source['source_file_sha256']=m.fingerprint(file)
    with pytest.raises(ValueError,match='Nonempty original flow'):
        m.read_source(file,source,chunk_size=2)

def test_raw_ip_not_allowed_even_when_column_exists():
    cols=['Flow ID',' Source IP',' Source Port',' Destination IP',
          ' Destination Port',' Protocol',' Timestamp',' Label']
    cols+=[' Feature '+str(i) for i in range(70)]
    selected=m.select_features(cols)
    assert len(selected)==71
    assert not any(m.normalize(x) in m.BANNED for x in selected)

def test_no_training_on_unknown_source(tmp_path):
    p=tmp_path/'UNSW_NB15_testing-set.csv'
    p.write_text('a,b\n1,2')
    with pytest.raises(ValueError):
        m.read_source(p,{'source_file':p.name,'encoding':'utf-8-sig'},chunk_size=1)

def test_validation_only_threshold_tie_break():
    y=np.array([0,0,1,1],dtype=np.uint8)
    p=np.array([0.2,0.4,0.6,0.8])
    selected=m.choose_threshold(y,p)
    assert selected['threshold']==pytest.approx(0.6,abs=1e-10)
    assert selected['validation_cost']==0
    assert selected['tie_break']=='largest_threshold'

def test_binary_metrics_and_low_support_labels():
    y=np.array([0,0,1,1],dtype=np.uint8)
    proba=np.array([0.01,0.1,0.9,0.8])
    labels=np.array(['BENIGN','BENIGN','DDoS','Heartbleed'])
    result=m.evaluate(y,proba,0.5,labels)
    assert result['macro_f1']==1.0
    assert result['tn']==2 and result['tp']==2
    assert result['unseen_family_recall']['Heartbleed']['support']==1
    assert result['unseen_family_recall']['Heartbleed']['low_support'] is True

def test_unseen_day_fold_constraints():
    def make(labels):
        ys=np.array([0 if s=='BENIGN' else 1 for s in labels],dtype=np.uint8)
        return {'X':np.arange(len(labels)*70).reshape(len(labels),70).astype('float32'),
                'y':ys,'labels':np.array(labels)}
    by={
     'Monday':make(['BENIGN']),
     'Tuesday':make(['BENIGN','FTP-Patator']),
     'Wednesday':make(['BENIGN','DoS Hulk']),
     'Thursday':make(['BENIGN','Web Attack – XSS']),
     'Friday':make(['BENIGN','DDoS'])
    }
    f={'train_days':['Monday','Thursday','Friday'],
        'validation_day':'Wednesday','test_day':'Tuesday',
        'unseen_test_classes':['FTP-Patator']}
    out=m.get_fold(by,f)
    assert out[0].shape[0]==5
    assert out[5].tolist()==[0,1]
    by['Monday']['labels']=np.array(['FTP-Patator'])
    by['Monday']['y']=np.array([1],dtype=np.uint8)
    with pytest.raises(AssertionError):
        m.get_fold(by,f)
