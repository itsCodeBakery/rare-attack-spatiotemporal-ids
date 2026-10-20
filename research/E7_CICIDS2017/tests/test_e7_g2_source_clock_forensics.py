"""Synthetic regression tests for E7 G2 structural padding and raw clock ordering."""
from pathlib import Path
import importlib.util
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('g2',ROOT/'e7_g2_source_clock_forensics.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_trailing_blank_rows_are_not_assumed_benign(tmp_path):
    p=tmp_path/'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    rows=[
      ['a','10.0.0.1','10.0.0.2','6/7/2017 8:59','BENIGN',100],
      ['b','10.0.0.3','10.0.0.2','6/7/2017 9:00','Web Attack – Brute Force',120],
      [None,None,None,None,None,None],
      ['  ','  ','  ','  ','  ','  '],
      [None,None,None,None,None,None],
    ]
    columns=['Flow ID',' Source IP',' Destination IP',' Timestamp',' Label',' Flow Duration']
    pd.DataFrame(rows,columns=columns).to_csv(p,index=False,encoding='cp1252')
    old=p.read_bytes()
    o=m.missing_block(p,encoding='cp1252',chunk_size=2)
    assert o['rows']==5
    assert o['counts']['no_label']==3
    assert o['counts']['no_timestamp']==3
    assert o['counts']['missing_label_and_all_fields_empty']==3
    assert o['counts']['missing_label_but_other_fields_present']==0
    assert o['first_missing_label_row_1based']==3
    assert o['last_missing_label_row_1based']==5
    assert o['missing_label_runs_in_csv_order']==1
    assert o['longest_missing_label_run']==3
    assert o['trailing_all_fields_empty_rows']==3
    assert o['nonempty_fields_per_source_row_histogram']['0']==3
    assert p.read_bytes()==old
    assert '10.0.0.1' not in str(o)
    assert o['training_authorized'] is False

def test_missing_label_but_real_traffic_features_present(tmp_path):
    p=tmp_path/'source.csv'
    pd.DataFrame({
     ' Source IP':['a','b'],
     ' Destination IP':['c','d'],
     ' Timestamp':['6/7/2017 8:59','6/7/2017 9:00'],
     ' Flow Duration':['10','15'],
     ' Label':['BENIGN',None],
    }).to_csv(p,index=False)
    o=m.missing_block(p,encoding='utf-8-sig',chunk_size=1)
    assert o['counts']['missing_label_but_other_fields_present']==1
    assert o['counts']['missing_label_and_all_fields_empty']==0

def test_12_to_1_and_other_timestamp_reversals_differ(tmp_path):
    p=tmp_path/'Monday-WorkingHours.pcap_ISCX.csv'
    pd.DataFrame({' Timestamp':[
      '03/07/2017 11:59:00','03/07/2017 12:00:00',
      '03/07/2017 01:00:00','03/07/2017 01:01:00',
      '03/07/2017 01:00:30','03/07/2017 01:00:30'
    ]}).to_csv(p,index=False)
    o=m.clock_order(p,encoding='utf-8-sig',timestamp_format='%d/%m/%Y %H:%M:%S',
                    expected_weekday='Monday',chunk_size=2)
    assert o['adjacent_reversals_total']==2
    assert o['adjacent_12_to_1_hour_reversals']==1
    assert o['adjacent_other_reversals']==1
    assert o['adjacent_tied_timestamps']==1
    assert o['causal_order_verified'] is False
    assert o['clock_correction_applied'] is False

def test_unsw_is_frozen_and_bad_encoding_not_accepted(tmp_path):
    p=tmp_path/'UNSW-NB15_testing-set.csv'
    p.write_text('a,b\n1,2')
    with pytest.raises(ValueError):m.missing_block(p,encoding='utf-8-sig')
    with pytest.raises(ValueError):m.clock_order(p,encoding='utf-8-sig',
       timestamp_format='%d/%m/%Y %H:%M',expected_weekday='Monday')
    p2=tmp_path/'some.csv'
    p2.write_text('a,b\n1,2')
    with pytest.raises(ValueError):m.missing_block(p2,encoding='latin1')
