"""Synthetic G0 public-archive security and idempotence tests; no real flows."""
import importlib.util
import json
from pathlib import Path
import pytest
import base64

module_path=Path(__file__).resolve().parents[1]/'scripts/e7_g0_publish.py'
spec=importlib.util.spec_from_file_location('e7_g0_publish',module_path)
pub=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

def sample():
    return {'phase':'E7_G0_READ_ONLY_DATA_PROVENANCE','external_source_basename':'day.csv',
       'input_artifact_sha256':'a'*64,'timestamp_format_declared':'%d/%m/%Y %H:%M',
       'full_file_scan':True,'not_trained':True,'unsw_official_test_touched':False,
       'temporal_causality_verified':False,'no_raw_endpoint_identifiers_exported':True,
       'files':[{'relative_csv_member':'day.csv','rows_inspected':10,'labels':{'Benign':10},
          'label_day_counts':{'2017-07-07 | Benign':10},'parse_failures':0,
          'timestamp_disorder_adjacent':0,'missing_endpoint_rows':0,
          'status':'SCHEMA_VALID_NOT_CAUSALITY_CERTIFIED',
          'original_column_names':['Timestamp','Source IP','Destination IP','Source Port','Destination Port','Protocol','Label'],
          'columns_resolved':dict(zip(('time','src','dst','sport','dport','protocol','label'),('Timestamp','Source IP','Destination IP','Source Port','Destination Port','Protocol','Label'))),
          'timestamp_parse_fraction':1.0}], 'total_rows_inspected':10,
       'status':'G0_SCHEMA_CHECKED_TEMPORAL_VALIDITY_PENDING',
       'explicit_followup_gates':['Verify causal flow start/end and session semantics']}

def test_allows_aggregate_only():
    assert pub.verify_report(sample())

def test_forbids_raw_endpoints():
    d=sample(); d['files'][0]['raw_ip']='1.2.3.4'
    with pytest.raises(ValueError,match='Unexpected'):pub.verify_report(d)

def test_forbids_sample_but_preserves_labelled_failure():
    d=sample(); d['full_file_scan']=False
    with pytest.raises(ValueError):pub.verify_report(d)
    d=sample();d['files'][0]['parse_failures']=2
    d['files'][0]['status']='DATA_QUALITY_REVIEW_REQUIRED'
    d['status']='G0_NOT_CLEARED'
    assert pub.verify_report(d)
    d['status']='G0_SCHEMA_CHECKED_TEMPORAL_VALIDITY_PENDING'
    with pytest.raises(ValueError):pub.verify_report(d)

def test_forbids_unsw_and_training():
    d=sample();d['unsw_official_test_touched']=True
    with pytest.raises(ValueError):pub.verify_report(d)
    d=sample();d['not_trained']=False
    with pytest.raises(ValueError):pub.verify_report(d)

def test_existing_different_record_fails(monkeypatch,tmp_path):
    p=tmp_path/'day.json';p.write_text(json.dumps(sample()))
    monkeypatch.setattr(pub,'github_json',lambda method,url,token,obj=None: {'content':base64.b64encode(b'wrong').decode()} )
    with pytest.raises(RuntimeError,match='refusing to overwrite'):
        pub.publish_report(p,token='fake',run_path=pub.PREFIX)

def test_existing_exact_record_does_not_reupload(monkeypatch,tmp_path):
    p=tmp_path/'day.json';p.write_text(json.dumps(sample()))
    data,_=pub.prepare_public_blob(p)
    monkeypatch.setattr(pub,'github_json',lambda method,url,token,obj=None: {'content':base64.b64encode(data).decode()} )
    result=pub.publish_report(p,token='fake',run_path=pub.PREFIX)
    assert result['status']=='ALREADY_SAVED_IDENTICALLY'
