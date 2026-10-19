"""E5 synthetic mechanism audit tests: no UNSW source/test data touched."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd

def test_label_blocking_run_lengths_vs_class_marginal_chance():
    from scripts.run_e5_context_diagnostic import (
        source_sequence_summary,window_purity
    )
    y=np.array([0]*6+[1]*5+[0]*4,dtype=np.int64)
    summary,transitions=source_sequence_summary(y,["A","B"],"synthetic")
    assert summary["num_label_runs"]==3
    assert summary["max_label_run_length"]==6
    assert summary["observed_adjacent_same_label_rate"]==12/14
    assert summary["independent_class_marginal_same_label_expectation"]==(
        (10/15)**2+(5/15)**2)
    assert len(transitions)==4 and sum(x["pair_count"] for x in transitions)==14
    purity=window_purity(y,["A","B"],"synthetic")
    assert purity[-1]["target_windows"]==11
    assert 0<purity[-1]["all_four_history_labels_match_target_fraction"]<1

def test_original_index_gaps_as_positions_not_time():
    from scripts.run_e5_context_diagnostic import position_gaps
    x=np.array([0,1,2,4,6,7,9,13])
    report=position_gaps(x,"synthetic")
    assert len(report)==4
    assert report[0]["history_position"]==1
    assert report[0]["target_windows"]==4
    assert report[0]["min_original_source_index_gap"]>=1
    assert all(0<=s["fully_contiguous_original_source_positions_fraction"]<=1
               for s in report)
    try:
        position_gaps([2,1,3,4,5],"synthetic")
        assert False,"Should reject unsorted frozen indices"
    except ValueError:pass

def test_historical_single_record_intervention_preserves_target():
    from scripts.run_e5_context_diagnostic import perturb_one_lag
    f=pd.DataFrame({
        "lag4__dur":[1.,2.,3.,4.,5.,6.,7.,8.],
        "lag4__proto":pd.Categorical(["tcp","udp"]*4),
        "lag0__dur":[11.,12.,13.,14.,15.,16.,17.,18.],
        "lag0__proto":pd.Categorical(["icmp","tcp"]*4)
    })
    changed=perturb_one_lag(f,["dur","proto"],4,9021)
    assert f["lag0__dur"].equals(changed["lag0__dur"])
    assert f["lag0__proto"].equals(changed["lag0__proto"])
    assert not changed["lag4__dur"].equals(f["lag4__dur"])
    assert set(changed["lag4__proto"].cat.categories)=={"tcp","udp"}
    np.testing.assert_array_equal(
        np.sort(f["lag4__dur"]),np.sort(changed["lag4__dur"]))

def test_aggregate_only_class_confusion_and_importance_groups():
    from scripts.run_e5_context_diagnostic import class_diagnostics,importance_group
    labels=np.arange(10,dtype=np.int64)
    prob=np.eye(10)*.8+.02
    prob[0]=.02
    prob[0,3]=.82
    result,cm,per,bins,wrong=class_diagnostics(labels,prob,[
        "Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic",
        "Normal","Reconnaissance","Shellcode","Worms"
    ])
    assert result["n"]==10
    assert cm[0,3]==1 and result["accuracy"]==.9
    assert len(per)==10 and len(bins)==100
    assert wrong[0]["true_class"]=="Analysis"
    assert importance_group("lag0__dur")=="current_record_lag0"
    assert importance_group("lag4__dur")=="historical_lag4"
    assert importance_group("stats__dur__five_std")=="engineered_window_statistics"

def test_no_new_model_or_official_data_requirements_and_notebook():
    cfg=json.loads(Path("configs/e5_context_diagnostic_001.json").read_text())
    assert cfg["official_test"]=="FORBIDDEN"
    assert cfg["feature_count"]==444
    assert cfg["internal_val_windows"]==35005
    assert cfg["counterfactual"]["repetitions"]==len(cfg["counterfactual"]["seeds"])==3
    for path in (
        "scripts/run_e5_context_diagnostic.py",
        "scripts/sync_e5_aggregate.py",
    ):
        ast.parse(Path(path).read_text())
    main=Path("scripts/run_e5_context_diagnostic.py").read_text()
    assert "lgb.Booster(model_file=" in main
    assert "model.predict(" in main
    assert "lgb.train(" not in main
    assert "official_test_predictions_PRIVATE" not in main
    assert "testing-set.csv" not in main
    assert "archive_window_e3" in main
    sync=Path("scripts/sync_e5_aggregate.py").read_text()
    assert "ALLOWLIST" in sync
    assert "summary.json" in sync
    assert '"trained_new_models"]!=0' in sync
    notebook=json.loads(Path("notebooks/22_e5_training_only_context_error_diagnostic.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":
            source="".join(cell["source"])
            ast.parse(source)
            assert "run_e5_context_diagnostic.py" in source
            assert "get_secret(\"pushRare\")" in source
            assert "run_official_001.py" not in source
