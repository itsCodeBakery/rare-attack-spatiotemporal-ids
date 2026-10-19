"""Synthetic reliability tests for 404-after-success GitHub archives and extended G0.

Never uses live GitHub or UNSW test. All token strings are fake.
"""
import base64
import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError
import pytest

STUDY=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('publisher_v2',STUDY/'scripts/e7_g0_publish_v2.py')
pub=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

def sample():
    return {
      'phase':'E7_G0_READ_ONLY_DATA_PROVENANCE','external_source_basename':'day.csv',
      'input_artifact_sha256':'a'*64,'timestamp_format_declared':'%d/%m/%Y %H:%M',
      'full_file_scan':True,'not_trained':True,'unsw_official_test_touched':False,
      'temporal_causality_verified':False,'no_raw_endpoint_identifiers_exported':True,
      'files':[{'relative_csv_member':'day.csv','rows_inspected':10,'labels':{'BENIGN':10},
        'label_day_counts':{'2017-07-07 | BENIGN':10},'parse_failures':0,
        'timestamp_disorder_adjacent':0,'missing_endpoint_rows':0,
        'status':'SCHEMA_VALID_NOT_CAUSALITY_CERTIFIED',
        'original_column_names':['Timestamp','Source IP','Destination IP','Source Port','Destination Port','Protocol','Label'],
        'columns_resolved':dict(zip(('time','src','dst','sport','dport','protocol','label'),
           ('Timestamp','Source IP','Destination IP','Source Port','Destination Port','Protocol','Label'))),
        'timestamp_parse_fraction':1.0}],
      'total_rows_inspected':10,'status':'G0_SCHEMA_CHECKED_TEMPORAL_VALIDITY_PENDING',
      'explicit_followup_gates':['Verify causal flow order']}
def http404(url):
    return HTTPError(url,404,'Not Found',None,None)

def test_after_put_success_but_contents_get_returns_404(monkeypatch,tmp_path):
    doc=sample(); f=tmp_path/'day.json';f.write_text(json.dumps(doc))
    data=pub.encode_safe(doc,'G0')
    calls={'put':0,'bad_get':0}
    def fake(method,url,token,payload=None):
        if method=='PUT':
            calls['put']+=1
            return {'content':{'sha':'example_blob'},'commit':{'sha':'example_commit'}}
        if '/git/blobs/' in url:
            return {'encoding':'base64','content':base64.b64encode(data).decode('ascii'),
                    'sha':'example_blob'}
        if '?ref=' in url: raise http404(url)
        if method=='GET':
            calls['bad_get']+=1
            raise http404(url)
        raise AssertionError(method)
    monkeypatch.setattr(pub,'api',fake)
    result=pub.publish(f,token='fake',stage='G0',receipts_dir=tmp_path,pause=lambda s:None)
    assert result['status']=='VERIFIED'
    assert result['verified_via']=='blob'
    assert calls['put']==1
    assert calls['bad_get']>=1
    receipt=list(tmp_path.glob('*.receipt.json'))
    assert len(receipt)==1
    assert json.loads(receipt[0].read_text())['status']=='VERIFIED'
    result2=pub.publish(f,token='fake',stage='G0',receipts_dir=tmp_path,pause=lambda s:None)
    assert result2['status']=='VERIFIED'
    assert calls['put']==1

def test_existing_remote_identical_needs_no_put(monkeypatch,tmp_path):
    f=tmp_path/'day.json';d=sample();f.write_text(json.dumps(d))
    content=pub.encode_safe(d,'G0')
    def fake(method,url,token,payload=None):
        assert method=='GET'
        return {'content':base64.b64encode(content).decode('ascii'),'sha':'good'}
    monkeypatch.setattr(pub,'api',fake)
    result=pub.publish(f,token='fake',stage='G0',receipts_dir=tmp_path,pause=lambda x:None)
    assert result['status']=='VERIFIED'

def test_existing_remote_mismatch_refuses_overwrite(monkeypatch,tmp_path):
    f=tmp_path/'day.json';f.write_text(json.dumps(sample()))
    monkeypatch.setattr(pub,'api',lambda method,url,token,payload=None:{
        'content':base64.b64encode(b'wrong').decode(),'sha':'wrong'})
    with pytest.raises(RuntimeError,match='differs'):
        pub.publish(f,token='fake',stage='G0',receipts_dir=tmp_path,pause=lambda x:None)

def test_pending_receipt_does_not_second_put(monkeypatch,tmp_path):
    f=tmp_path/'day.json';d=sample();f.write_text(json.dumps(d))
    raw=pub.encode_safe(d,'G0')
    digest=__import__('hashlib').sha256(raw).hexdigest()
    remote=pub.PREFIX+'/G0/day.csv.audit.json'
    receipt=tmp_path/'G0__day.csv.audit.json.receipt.json'
    pub.atomic_json(receipt,{'remote':remote,'source_sha256':digest,
                    'blob_sha':'sha_of_existing_put','commit_sha':'commit_of_existing_put',
                    'status':'PUT_ACCEPTED_VERIFY_PENDING'})
    calls={'put':0}
    def missing(method,url,token,payload=None):
        if method=='PUT':calls['put']+=1
        raise http404(url)
    monkeypatch.setattr(pub,'api',missing)
    with pytest.raises(RuntimeError,match='receipt exists'):
        pub.publish(f,token='fake',stage='G0',receipts_dir=tmp_path,pause=lambda x:None)
    assert calls['put']==0

def extended():
    return {'schema':'E7_G0_EXTENDED_AGGREGATE_001','status':'REVIEW_REQUIRED',
      'source_file':'day.csv','source_sha256':'b'*64,'rows':5,
      'labels':{'BENIGN':5},
      'timestamp_quality':{'raw_hour_histogram':{'1':5},'parsed_weekdays':{'Friday':5},
        'parse_failures':0,'row_order_reversals':0,'timestamp_min':'2017-07-07T01:00:00',
        'timestamp_max':'2017-07-07T01:10:00','clock_ambiguous':True},
      'duration_quality':{'duration_unit':'UNVERIFIED_ORIGINAL_FLOW_DURATION_UNITS',
        'duration_zero':0,'duration_negative':0,'duration_missing':0,
        'duration_nonfinite':0,'duration_max_raw':250.0,'duration_min_raw':2.0},
      'feature_quality':{'numeric_feature_count':2,
        'columns_with_missing':{},'columns_with_nonfinite':{},
        'columns_with_negative':{},'columns_with_zero':{},
        'constant_numeric_columns':[],'missing_endpoint_rows':0},
      'duplicates':{'candidate_duplicate_rows':0,'candidate_label_conflict_rows':0,
        'hash_scheme':'64-bit candidate','disclaimer':'hash collision possible'},
      'notes':['No model fitted']}

def test_extended_aggregate_schema_and_raw_keys_rejected():
    assert pub.encode_safe(extended(),'G0_EXTENDED')
    d=extended();d['feature_quality']['ip_address']='127.0.0.1'
    with pytest.raises(ValueError):pub.encode_safe(d,'G0_EXTENDED')

def test_extended_aggregated_labels_cannot_be_rows():
    d=extended();d['labels']={'BENIGN':['raw','rows']}
    with pytest.raises(ValueError):pub.encode_safe(d,'G0_EXTENDED')
