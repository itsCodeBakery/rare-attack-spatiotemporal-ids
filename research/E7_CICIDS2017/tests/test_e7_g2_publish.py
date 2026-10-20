"""Synthetic E7 G2 GitHub publisher regressions: transient 404 and no overwrite."""
from pathlib import Path
import base64
import hashlib
import importlib.util
import json
from urllib.error import HTTPError
import pytest

ROOT=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('g2_publish',ROOT/'e7_g2_publish.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def report():
    return {
        'schema':'E7_G2_TIMESTAMP_ORDER_FORENSICS_001',
        'source_file':'Monday-WorkingHours.pcap_ISCX.csv',
        'source_sha256':'a'*64,'source_encoding':'utf-8-sig',
        'declared_time_format':'%d/%m/%Y %H:%M:%S','expected_weekday':'Monday',
        'rows':6,'parse_failures':0,
        'raw_hour_distribution':{'01':2,'12':1},
        'parsed_weekday_distribution':{'Monday':6},
        'adjacent_reversals_total':2,
        'adjacent_12_to_1_hour_reversals':1,
        'adjacent_other_reversals':1,
        'adjacent_tied_timestamps':0,'adjacent_forward_steps':3,
        'adjacent_reversal_magnitude_buckets':{},
        'weekday_consistent':True,
        'source_modified':False,'clock_correction_applied':False,
        'causal_order_verified':False,'training_authorized':False,
        'interpretation':'Aggregate diagnostic only'
    }

def not_found(url):
    return HTTPError(url,404,'Not Found',None,None)

def test_put_succeeds_even_if_immediate_contents_get_404(monkeypatch,tmp_path):
    d=report();raw=m.canonical_bytes(d)
    calls={'put':0}
    def mock(method,url,token,body=None):
        if method=='PUT':
            calls['put']+=1
            return {'content':{'sha':'new_blob'},'commit':{'sha':'new_commit'}}
        if '/git/blobs/' in url:
            return {'content':base64.b64encode(raw).decode('ascii')}
        raise not_found(url)
    monkeypatch.setattr(m,'api',mock)
    out=m.publish(d,token='FAKE',local_dir=tmp_path,pause=lambda _:None)
    assert out['status']=='UPLOADED_VERIFIED'
    assert calls['put']==1
    assert json.loads(next(tmp_path.glob('*.receipt.json')).read_text())['verified_via']=='blob'
    result=m.publish(d,token='FAKE',local_dir=tmp_path,pause=lambda _:None)
    assert result['status']=='VERIFIED'
    assert calls['put']==1

def test_mismatched_github_report_never_overwritten(monkeypatch,tmp_path):
    def mock(method,url,token,body=None):
        if method=='PUT':raise AssertionError('Would overwrite original')
        return {'content':base64.b64encode(b'WRONG').decode('ascii')}
    monkeypatch.setattr(m,'api',mock)
    with pytest.raises(RuntimeError,match='conflicting'):
        m.publish(report(),token='FAKE',local_dir=tmp_path,pause=lambda _:None)

def test_existing_put_receipt_blocks_second_put(monkeypatch,tmp_path):
    d=report();raw=m.canonical_bytes(d)
    remote=m.PREFIX+'/'+d['source_file']+'.g2.json'
    receipt=tmp_path/(d['source_file']+'.g2.json.receipt.json')
    m.write_receipt(receipt,{'remote':remote,'sha256':hashlib.sha256(raw).hexdigest(),
                              'blob':'prior_blob','commit':'prior_commit'})
    def mock(method,url,token,body=None):
        if method=='PUT':raise AssertionError('Duplicate PUT prohibited')
        raise not_found(url)
    monkeypatch.setattr(m,'api',mock)
    with pytest.raises(RuntimeError,match='no duplicate'):
        m.publish(d,token='FAKE',local_dir=tmp_path,pause=lambda _:None)

def test_raw_field_and_training_rejected():
    d=report();d['source_ip']='1.2.3.4'
    with pytest.raises(ValueError):m.canonical_bytes(d)
    d=report();d['training_authorized']=True
    with pytest.raises(ValueError):m.canonical_bytes(d)
