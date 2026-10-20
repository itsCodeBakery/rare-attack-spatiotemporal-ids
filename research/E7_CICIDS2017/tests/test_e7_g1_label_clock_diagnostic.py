"""G1 regression: missing-label source runs and timestamp no-causality report."""
import importlib.util
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("e7_g1",ROOT/"scripts/e7_g1_label_clock_diagnostic.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_missing_label_runs_are_not_reclassified(tmp_path):
    p=tmp_path/"Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    rows=[
       {" Timestamp":"6/7/2017 8:59"," Label":"BENIGN"},
       {" Timestamp":"6/7/2017 8:59"," Label":None},
       {" Timestamp":"6/7/2017 9:00"," Label":None},
       {" Timestamp":"6/7/2017 9:00"," Label":"Web Attack – XSS"},
       {" Timestamp":"6/7/2017 9:01"," Label":None},
    ]
    pd.DataFrame(rows).to_csv(p,index=False)
    raw=p.read_bytes()
    result=m.inspect(p,timestamp_format="%d/%m/%Y %H:%M",expected_day="Thursday",chunksize=2)
    assert result["rows"]==5
    assert result["missing_label_rows"]==3
    assert result["missing_label_runs_in_csv_row_order"]==2
    assert result["longest_missing_label_run_in_csv_row_order"]==2
    assert result["labels"]["__MISSING_LABEL__"]==3
    assert result["first_missing_row_1based"]==2
    assert result["last_missing_row_1based"]==5
    assert result["training_authorized"] is False
    assert result["causal_order_proven"] is False
    assert p.read_bytes()==raw
