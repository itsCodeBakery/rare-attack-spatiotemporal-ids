"""Synthetic original-byte encoding regression tests for overnight E7 resume."""
import importlib.util
import json
from pathlib import Path
import pytest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def load(script,asname):
    spec=importlib.util.spec_from_file_location(asname,ROOT/'scripts'/script)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

gate=load('e7_csv_encoding_gate.py','e7_encoding_gate')
basic=load('e7_g0_external_source_audit_v2.py','e7_source_audit_v2')
quality=load('e7_g0_extended_audit_v2.py','e7_quality_audit_v2')
pub=load('e7_g0_publish_v2.py','e7_publisher_v2_for_encoding')

HEADERS=['Flow ID',' Source IP',' Source Port',' Destination IP',' Destination Port',
         ' Protocol',' Timestamp',' Flow Duration',' Total Fwd Packets',' Label']

def fixture(tmp_path,*,codec='cp1252',non_utf8=True):
    p=tmp_path/'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    header=','.join(HEADERS)+'\n'
    row1=['a','1.2.3.4','1234','10.2.3.4','80','6','6/7/2017 8:59','100','7','BENIGN']
    row2=['b','1.2.3.5','1235','10.2.3.5','80','6','6/7/2017 9:00','200','9','Web Attack – Brute Force' if non_utf8 else 'Web Attack - Brute Force']
    content=header+','.join(row1)+'\n'+','.join(row2)+'\n'
    p.write_bytes(content.encode(codec))
    return p

def test_windows_1252_dash_0x96_source_is_never_replaced(tmp_path):
    f=fixture(tmp_path)
    raw=f.read_bytes()
    assert b'\x96' in raw
    result=gate.source_encoding_gate(f,block_size=257)
    assert result['source_encoding']=='cp1252'
    assert result['requires_manual_review']
    info=basic.inspect(f,timestamp_format='%d/%m/%Y %H:%M',
                       source_encoding=result['source_encoding'],chunk_size=1)
    assert info['total_rows_inspected']==2
    assert info['status']=='G0_NOT_CLEARED'
    assert any('ENCODING_REVIEW_REQUIRED' in x for x in info['explicit_followup_gates'])
    assert 'Web Attack – Brute Force' in info['files'][0]['labels']
    assert pub.encode_safe(info,'G0')
    q=quality.audit_file(f,timestamp_format='%d/%m/%Y %H:%M',
                  expected_weekday='Thursday',source_encoding='cp1252',chunk_size=1)
    assert q['labels']['Web Attack – Brute Force']==1
    assert q['status']=='REVIEW_REQUIRED'
    assert pub.encode_safe(q,'G0_EXTENDED')
    assert f.read_bytes()==raw
    assert q['source_sha256']==info['input_artifact_sha256']==result['original_csv_sha256']

def test_plain_utf8_source_follows_original_path(tmp_path):
    f=fixture(tmp_path,codec='utf-8',non_utf8=False)
    result=gate.source_encoding_gate(f,block_size=257)
    assert result['source_encoding']=='utf-8-sig'
    assert not result['requires_manual_review']
    info=basic.inspect(f,timestamp_format='%d/%m/%Y %H:%M',
                       source_encoding='utf-8-sig',chunk_size=1)
    assert info['status']=='G0_SCHEMA_CHECKED_TEMPORAL_VALIDITY_PENDING'

def test_mixed_bom_invalid_utf8_fails_closed(tmp_path):
    f=fixture(tmp_path)
    f.write_bytes(b'\xef\xbb\xbf'+f.read_bytes())
    with pytest.raises(ValueError,match='BOM_MIXED_ENCODING'):
        gate.source_encoding_gate(f)

def test_invalid_cp1252_byte_fails_closed(tmp_path):
    f=fixture(tmp_path)
    f.write_bytes(f.read_bytes()+b'\x81')
    with pytest.raises(ValueError,match='ENCODING_UNRESOLVED'):
        gate.source_encoding_gate(f)

def test_rejects_lossy_bytes_instead_of_silent_replacement(tmp_path):
    f=fixture(tmp_path)
    data=f.read_bytes()
    with pytest.raises(UnicodeDecodeError):data.decode('utf-8')
    assert f.read_bytes()==data
