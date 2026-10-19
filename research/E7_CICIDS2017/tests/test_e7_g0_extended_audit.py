"""Small synthetic source scan proves no input writes, no raw IP/record output."""
import importlib.util
import json
from pathlib import Path
import pandas as pd

STUDY=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('extended_scan',STUDY/'scripts/e7_g0_extended_audit.py')
scan=importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

def test_extended_publisher_safe_aggregate_and_candidate_duplicates(tmp_path):
    p=tmp_path/'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    data=pd.DataFrame([
      ['a','1.2.3.4',1234,'10.0.0.1',80,6,'7/7/2017 3:30',10,5,'BENIGN'],
      ['a','1.2.3.4',1234,'10.0.0.1',80,6,'7/7/2017 3:30',10,5,'BENIGN'],
      ['a','1.2.3.4',1234,'10.0.0.1',80,6,'7/7/2017 3:30',10,5,'DDoS'],
      ['b','1.2.3.6',0,'10.0.0.9',80,6,'7/7/2017 3:31',-1,0,'DDoS'],
    ],columns=['Flow ID',' Source IP',' Source Port',' Destination IP',
               ' Destination Port',' Protocol',' Timestamp',' Flow Duration',
               ' Fwd Packet Length Max',' Label'])
    data.to_csv(p,index=False)
    before=p.read_bytes()
    r=scan.audit_file(p,timestamp_format='%d/%m/%Y %H:%M',
                      expected_weekday='Friday',chunk_size=2)
    assert before==p.read_bytes()
    assert r['rows']==4
    assert r['labels']=={'BENIGN':2,'DDoS':2}
    assert r['duplicates']['candidate_duplicate_rows']==2
    assert r['duplicates']['candidate_label_conflict_rows']==1
    assert r['duration_quality']['duration_negative']==1
    assert r['timestamp_quality']['clock_ambiguous'] is True
    text=json.dumps(r)
    assert '1.2.3.4' not in text and '10.0.0.1' not in text
    assert 'E7_G0_EXTENDED_AGGREGATE_001' in text
    assert r['status']=='REVIEW_REQUIRED'

def test_never_accesses_unsw_test(tmp_path):
    p=tmp_path/'UNSW_NB15_testing-set.csv'
    p.write_text('Timestamp,Label\nfake,0')
    try:scan.audit_file(p,timestamp_format='%d/%m/%Y %H:%M',expected_weekday='Monday')
    except ValueError as exc:assert 'UNSW' in str(exc)
    else:raise AssertionError('UNSW test was not blocked')
