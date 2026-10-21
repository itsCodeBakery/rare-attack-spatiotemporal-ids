"""Exact G3 private preflight archive recovery regression."""
import io
import zipfile
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location(
    "preflight_v3",Path(__file__).resolve().parents[1]/"scripts/e7_g3_archive_v3.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_preflight_v3_repeatable(monkeypatch,tmp_path):
    received=[]
    def capture(data,**kwargs):
        received.append((data,kwargs["asset_name"]))
        return "PRIVATE_EXISTING_VERIFIED"
    monkeypatch.setattr(m,"archive_model",capture)
    for _ in range(3):
        assert m.preflight_private_backups("FAKE",{"draft":True},tmp_path)=="PRIVATE_EXISTING_VERIFIED"
    assert len(received)==3
    assert len({d for d,n in received})==1
    assert all(n=="G3_BACKUP_PREFLIGHT_V3_PRIVATE.zip" for d,n in received)
    with zipfile.ZipFile(io.BytesIO(received[0][0])) as z:
        assert z.namelist()==["G3-backup-preflight.txt"]
        assert z.getinfo("G3-backup-preflight.txt").date_time==(1980,1,1,0,0,0)
