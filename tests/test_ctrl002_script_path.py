"""Regression: Kaggle invokes the V2 finalizer by absolute script filename.

In this mode sys.path[0] points to scripts/, not to the repo root.
The plotter must import V1 plotting helpers successfully without modifying the
research training environment or requiring any checkpoint.
"""
import os
from pathlib import Path
import subprocess
import sys


def test_v2_finalizer_help_imports_under_kaggle_script_path():
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy()
    env["PYTHONPATH"]=str(root/"src")
    result=subprocess.run(
        [sys.executable,str(root/"scripts"/"finalize_ctrl002_cnnvit.py"),"--help"],
        cwd=root,env=env,check=False,text=True,capture_output=True,timeout=40
    )
    assert result.returncode==0, result.stdout+"\n"+result.stderr
    assert "--checkpoints" in result.stdout
    assert "--reference" in result.stdout
