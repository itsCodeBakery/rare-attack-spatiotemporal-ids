"""G4 report schema, privacy and immutable GitHub publication safety tests."""
import importlib.util
import base64
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location(
    "g4pub",Path(__file__).resolve().parents[1]/"scripts/e7_g4_publish.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def record():
    sel={'alarm_high':1.000001,'auto_benign_low':-0.000001}
    return {'schema':'E7_G4_CALIBRATION_REPORT_001',
      'experiment':'E7-G4-RISK-ABSTENTION-001',
      'parent_experiment':'E7-G3-OPENWORLD-DAYHOLDOUT-001',
      'g4_protocol_git_blob':'a'*40,'g3_protocol_git_blob':'b'*40,
      'fold':'OPENWORLD_DAY_FOLD_1','seed':42,
      'train_days':['Monday','Thursday','Friday'],
      'validation_day':'Wednesday','test_day':'Tuesday',
      'original_source_hashes':{'f'+str(i):'f'*64 for i in range(8)},
      'original_model_sha256':'c'*64,'features_sha256':'d'*64,
      'original_g3_metrics_blob':'e'*40,
      'original_label_provenance':'UNVERIFIED_ORIGINAL_CICIDS2017_LABELS',
      'validation_only_selection':sel,
      'm0_g3_b1_macro_f1':0.4,
      'm1_alarm_metrics':{'rows':100,'fpr':0.01,'macro_f1':0.5},
      'm2_selective_metrics':{'rows':100,'coverage':0.5,'selective_error':0.1},
      'scientific_status':'RETROSPECTIVE_EXPLORATORY_REUSED_G3_TEST',
      'limitations':['already inspected test','source labels provisional','no transfer guarantee']}

def test_valid_schema_only_safe_aggregates():
    assert m.validate(record())
    r=record();r['source_ip']='192.0.2.3'
    with pytest.raises(ValueError):m.validate(r)
    r=record();r['g4_protocol_git_blob']='f'*64
    with pytest.raises(AssertionError):m.validate(r)
    r=record();r['scientific_status']='CONFIRMATORY'
    with pytest.raises(AssertionError):m.validate(r)

def test_conflicting_public_result_never_overwritten(tmp_path):
    def api(method,url,token,body=None):
        assert method!='PUT'
        return {'content':base64.b64encode(b'WRONG').decode()}
    with pytest.raises(RuntimeError,match='Conflicting published'):
        m.publish(record(),api=api,token='FAKE',output_dir=tmp_path)

def test_existing_public_result_verifies_without_put(tmp_path):
    data=m.validate(record())
    def api(method,url,token,body=None):
        assert method=='GET'
        return {'content':base64.b64encode(data).decode()}
    assert m.publish(record(),api=api,token='FAKE',
                     output_dir=tmp_path)=='G4_VERIFIED_EXISTING'

def test_summary_frozen_counts():
    summary={'schema':'E7_G4_SUMMARY_001','experiment':'E7-G4-RISK-ABSTENTION-001',
       'g4_protocol_git_blob':'a'*40,'n_reports':20,'n_folds':4,
       'n_seeds':5,'fold_results':{
           'OPENWORLD_DAY_FOLD_'+str(i):{'test_day':'Tuesday'}
           for i in range(1,5)},
       'scientific_status':'RETROSPECTIVE_EXPLORATORY_REUSED_G3_TEST',
       'limitations':['Already inspected original G3 test data']}
    assert m.validate_summary(summary)
    summary['n_reports']=19
    with pytest.raises(AssertionError):m.validate_summary(summary)
