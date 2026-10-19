"""Synthetic-only E7 source-gate tests. Never read original benchmark traffic."""
import importlib.util
from pathlib import Path
import json
import zipfile
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'e7_g0_external_source_audit.py'
spec = importlib.util.spec_from_file_location('e7_g0', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
CSV = ('Timestamp,Source IP,Destination IP,Source Port,Destination Port,Protocol,Label,Flow Duration\n'
       '01/01/2024 10:00:00,1.1.1.1,2.2.2.2,123,80,6,BENIGN,100\n'
       '01/01/2024 10:01:00,1.1.1.1,2.2.2.2,123,80,6,DoS,120\n')

def test_explicit_timestamp_and_no_raw_endpoints(tmp_path):
    source = tmp_path / 'flows.csv'
    source.write_text(CSV)
    report = audit.inspect(source, timestamp_format='%d/%m/%Y %H:%M:%S')
    assert report['status']=='G0_SCHEMA_CHECKED_TEMPORAL_VALIDITY_PENDING'
    assert report['temporal_causality_verified'] is False
    assert report['total_rows_inspected']==2
    assert '1.1.1.1' not in json.dumps(report)
    assert report['files'][0]['labels']=={'BENIGN':1,'DoS':1}

def test_zip_nested_csv_and_sample_only(tmp_path):
    source = tmp_path / 'GeneratedLabelledFlows.zip'
    with zipfile.ZipFile(source,'w') as archive:
        archive.writestr('Wednesday/flows.csv', CSV)
    report = audit.inspect(source, timestamp_format='%d/%m/%Y %H:%M:%S', max_rows=1)
    assert report['status']=='G0_NOT_CLEARED'
    assert report['total_rows_inspected']==1

def test_missing_metadata_fails_closed(tmp_path):
    source = tmp_path / 'flows.csv'
    source.write_text('Timestamp,Label\n01/01/2024 10:00:00,normal\n')
    with pytest.raises(ValueError,match='missing required fields'):
        audit.inspect(source, timestamp_format='%d/%m/%Y %H:%M:%S')

def test_unsw_test_rejected(tmp_path):
    source = tmp_path / 'UNSW_NB15_testing-set.csv'
    source.write_text(CSV)
    with pytest.raises(ValueError,match='only selected EXTERNAL'):
        audit.inspect(source, timestamp_format='%d/%m/%Y %H:%M:%S')

def test_no_automatic_timestamp_assumptions(tmp_path):
    source = tmp_path / 'flows.csv'
    source.write_text(CSV)
    with pytest.raises(ValueError,match='Explicit'):
        audit.inspect(source, timestamp_format=None)

def test_bad_timestamp_never_passes(tmp_path):
    source = tmp_path / 'flows.csv'
    source.write_text(CSV.replace('01/01/2024 10:01:00','not-a-date'))
    report = audit.inspect(source, timestamp_format='%d/%m/%Y %H:%M:%S')
    assert report['status']=='G0_NOT_CLEARED'
    assert report['files'][0]['parse_failures']==1
