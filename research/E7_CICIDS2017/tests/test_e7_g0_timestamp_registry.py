"""Synthetic regression tests for 26A -> 26B timestamp registry (no dataset needed)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'configs/e7_g0_timestamp_registry_001.json'

SAMPLES = {
  'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv': '7/7/2017 3:30',
  'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv': '7/7/2017 1:00',
  'Friday-WorkingHours-Morning.pcap_ISCX.csv': '7/7/2017 8:59',
  'Monday-WorkingHours.pcap_ISCX.csv': '03/07/2017 08:55:58',
  'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv': '6/7/2017 1:00',
  'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv': '6/7/2017 8:59',
  'Tuesday-WorkingHours.pcap_ISCX.csv': '4/7/2017 8:54',
  'Wednesday-workingHours.pcap_ISCX.csv': '5/7/2017 8:42'
}

def registry():
    return json.loads(REGISTRY.read_text(encoding='utf-8'))

def test_exact_eight_files_and_preserved_path():
    cfg=registry()
    assert cfg['required_csv_count']==8
    assert cfg['actual_csv_parent_basename']=='TrafficLabelling '
    assert set(cfg['filename_registry'])==set(SAMPLES)
    assert cfg['timestamp_header_original']==' Timestamp'

def test_source_sample_weekdays_and_explicit_parsers():
    cfg=registry()
    for basename,text in SAMPLES.items():
        entry=cfg['filename_registry'][basename]
        value=pd.to_datetime([text],format=entry['timestamp_format'],errors='coerce',exact=True)
        assert not pd.isna(value[0]),basename
        assert value[0].day_name()==entry['expected_weekday'],basename

def test_monday_is_seconds_all_other_entries_are_minutes():
    cfg=registry()
    for name,entry in cfg['filename_registry'].items():
        expected='%d/%m/%Y %H:%M:%S' if name.startswith('Monday') else '%d/%m/%Y %H:%M'
        assert entry['timestamp_format']==expected,name

def test_no_automatic_assumption_or_pass_claim():
    cfg=registry()
    assert cfg['status'].endswith('FULL_SCAN_NOT_PERFORMED')
    assert 'SOURCE_CSV_BYTES_SHA256_REQUIRED' in cfg['guards']
    assert 'NO_UNSW_OFFICIAL_TEST_ACCESS' in cfg['guards']
