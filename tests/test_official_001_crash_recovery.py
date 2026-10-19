"""Recovery of one frozen official evaluation from saved evidence, no CSV access."""
import ast,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from scripts.recover_official_001 import recover

def dump(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True)+"\n")

def build_prior_failed_run(folder):
    n=82328
    model=folder/"final_model_PRIVATE.txt"
    model.write_text("SYNTHETIC DUMMY CHECKPOINT FOR TESTS ONLY")
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    test_prediction=np.zeros(n,dtype=np.uint8)
    probs=np.zeros((n,10),dtype=np.float32)
    probs[:,0]=1.0
    pred_file=folder/"official_test_predictions_PRIVATE.npz"
    np.savez_compressed(pred_file,y_true=test_prediction,
                        y_pred=test_prediction,
                        probabilities=probs,
                        official_test_row_indices=np.arange(4,82332,dtype=np.int32))
    classes=["Analysis","Backdoor","DoS","Exploits","Fuzzers",
             "Generic","Normal","Reconnaissance","Shellcode","Worms"]
    cm=confusion_matrix(test_prediction,test_prediction,labels=np.arange(10))
    pd.DataFrame(cm,index=classes,columns=classes).rename_axis(
        "actual_class").to_csv(folder/"official_confusion.csv")
    pd.DataFrame({"class":classes,"support":[n]+[0]*9,
                  "precision":[1.]+[0.]*9,"recall":[1.]+[0.]*9,
                  "f1":[1.]+[0.]*9}).to_csv(folder/"official_per_class.csv",index=False)
    config={"experiment_id":"WINDOW-OFFICIAL-001-S42",
            "status":"PRE_REGISTERED_BEFORE_OPENING_TEST",
            "final_train_iterations":58,
            "official_test_window_count":n,
            "original_testing_rows":82332,
            "paper_comparator":{"accuracy":0.8577,"macro_f1_reported":0.4687}}
    dump(folder/"config_effective.json",config)
    dump(folder/"PRETEST_FROZEN_CHECKPOINT.json",{
        "experiment_id":config["experiment_id"],
        "fixed_rounds":58,"official_test_opened":False,
        "selected_before_test":True,
        "model_checkpoint_sha256":sha(model)})
    dump(folder/"data_lineage.json",{
        "class_names":classes,
        "official_test_opened_before_checkpoint":False})
    report={
        "official_test_used":True,"used_for_training":False,
        "used_for_tuning":False,"model_locked_before_test":True,
        "model_checkpoint_sha256":sha(model),
        "official_predictions_sha256":sha(pred_file),
        "n":n,"evaluation_window_count":n,
        "fixed_training_rounds":58,
        "test_csv_sha256":"simulated_sha_not_real_data",
        "accuracy":1.0,"macro_f1":0.1,"weighted_f1":1.0}
    dump(folder/"official_metrics.json",report)
    (folder/"source_code_commit.txt").write_text("SYNTHETIC\n")
    (folder/"training_category_vocab_PRIVATE.json").write_text("{}\n")
    return sha(pred_file),sha(model)

def test_recovery_is_idempotent_reuses_previous_npz_never_tests_again(tmp_path):
    input_prediction_sha,model_sha=build_prior_failed_run(tmp_path)
    first=recover(tmp_path,False,None)
    second=recover(tmp_path,False,None)
    assert first==second
    assert hashlib.sha256((tmp_path/"official_test_predictions_PRIVATE.npz").read_bytes()).hexdigest()==input_prediction_sha
    assert hashlib.sha256((tmp_path/"final_model_PRIVATE.txt").read_bytes()).hexdigest()==model_sha
    comparison=json.loads((tmp_path/"paper_comparison.json").read_text())
    assert comparison["numeric_accuracy_difference_pp"]==100*(1-.8577)
    assert comparison["numeric_macro_f1_difference_pp"]==100*(.1-.4687)
    assert comparison["recovered_from_existing_original_predictions_without_second_test"] is True
    completion=json.loads((tmp_path/"official_evaluation_complete.json").read_text())
    assert completion["official_test_evaluated_once"] is True

def test_guard_refuses_original_training_repetition_and_recovery_never_reads_csv():
    original=Path("scripts/run_official_001.py").read_text()
    recovery=Path("scripts/recover_official_001.py").read_text()
    assert 'cfg["paper_comparator"]["macro_f1_reported"]' in original
    assert "official_test_predictions_PRIVATE.npz" in original.split("    # Only TRAINING input content")[0]
    assert "scripts/recover_official_001.py" in original
    assert "pd.read_csv" not in recovery
    assert "lgb.train" not in recovery
    assert "model.predict" not in recovery
    assert "from sklearn.metrics" in recovery
    ast.parse(recovery)
    nb=json.loads(Path("notebooks/21_official_001_recovery_NO_RETRAIN_NO_RETEST.ipynb").read_text())
    for cell in nb["cells"]:
        if cell["cell_type"]=="code":
            source="".join(cell["source"])
            ast.parse(source)
            assert "recover_official_001.py" in source
            assert "run_official_001.py" not in source
            assert "get_secret(\"pushRare\")" in source
