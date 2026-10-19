"""Post-crash recovery for ORIGINAL frozen official-test evaluation.

The initial one-time official test evaluation already wrote actual predictions,
metrics, class report and frozen model before an incorrect comparator dict key
raised KeyError. This script NEVER opens the UNSW testing (or training) CSV,
NEVER fits/predicts with a model, and NEVER changes saved probabilities.
It only verifies immutable local evidence, finalizes the missing reports, and
privately archives original results. Run in the SAME Kaggle runtime.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

if __package__:
    from scripts.archive_official_001 import archive_final, audit_test, sha
else:
    from archive_official_001 import archive_final, audit_test, sha

EXPECTED_EXPERIMENT="WINDOW-OFFICIAL-001-S42"

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def dump(path,data):
    contents=json.dumps(data,indent=2,sort_keys=True)+"\n"
    if path.exists():
        if load_json(path)!=data:
            raise ValueError(f"Existing immutable recovery record differs: {path.name}")
        return
    path.write_text(contents,encoding="utf-8")

def recover(output,do_archive,token):
    out=Path(output)
    mandatory=(
      "config_effective.json",
      "PRETEST_FROZEN_CHECKPOINT.json",
      "data_lineage.json",
      "source_code_commit.txt",
      "final_model_PRIVATE.txt",
      "training_category_vocab_PRIVATE.json",
      "official_metrics.json",
      "official_test_predictions_PRIVATE.npz",
      "official_confusion.csv",
      "official_per_class.csv",
    )
    missing=[p for p in mandatory if not (out/p).is_file()]
    if missing:
        raise FileNotFoundError(
            "No further official-test inference or refit permitted. "
            "The original Kaggle files must be recovered from the same session "
            "or a saved Kaggle output before finalizing: "+", ".join(missing))
    cfg=load_json(out/"config_effective.json")
    pretest=load_json(out/"PRETEST_FROZEN_CHECKPOINT.json")
    original=load_json(out/"official_metrics.json")
    lineage=load_json(out/"data_lineage.json")
    if (cfg["experiment_id"]!=EXPECTED_EXPERIMENT or
        pretest["experiment_id"]!=EXPECTED_EXPERIMENT or
        cfg["status"]!="PRE_REGISTERED_BEFORE_OPENING_TEST" or
        cfg["final_train_iterations"]!=58 or
        pretest["fixed_rounds"]!=58 or
        pretest["official_test_opened"] is not False or
        pretest["selected_before_test"] is not True or
        original["official_test_used"] is not True or
        original["used_for_training"] is not False or
        original["used_for_tuning"] is not False or
        lineage["official_test_opened_before_checkpoint"] is not False or
        original["model_locked_before_test"] is not True):
        raise ValueError("Saved experiment does not match frozen-before-test protocol")
    model_sha=sha(out/"final_model_PRIVATE.txt")
    prob_sha=sha(out/"official_test_predictions_PRIVATE.npz")
    if (model_sha!=pretest["model_checkpoint_sha256"] or
        model_sha!=original["model_checkpoint_sha256"] or
        prob_sha!=original["official_predictions_sha256"]):
        raise ValueError("Saved pretest checkpoint/prediction checksum mismatch")
    if (original["n"]!=cfg["official_test_window_count"] or
        original["evaluation_window_count"]!=cfg["official_test_window_count"] or
        original["fixed_training_rounds"]!=cfg["final_train_iterations"]):
        raise ValueError("Official evaluation targets or rounds unexpectedly changed")

    # Recompute summary from EXISTING SAVED labels/predictions, no test CSV read.
    pred_path=out/"official_test_predictions_PRIVATE.npz"
    with np.load(pred_path,allow_pickle=False) as src:
        y=src["y_true"].astype(np.int64)
        predicted=src["y_pred"].astype(np.int64)
        probabilities=src["probabilities"].astype(np.float64)
        indices=src["official_test_row_indices"].astype(np.int64)
    if (y.shape!=(82328,) or predicted.shape!=(82328,) or
        probabilities.shape!=(82328,10) or
        not np.array_equal(indices,np.arange(4,82332)) or
        not np.isfinite(probabilities).all() or
        np.any(y>9) or np.any(predicted>9) or
        not np.array_equal(np.argmax(probabilities,axis=1),predicted)):
        raise ValueError("Saved official test evidence has invalid shape/values")
    cm=confusion_matrix(y,predicted,labels=np.arange(10))
    acc=float(np.trace(cm)/len(y))
    macro=float(f1_score(y,predicted,labels=np.arange(10),
                         average="macro",zero_division=0))
    weighted=float(f1_score(y,predicted,labels=np.arange(10),
                            average="weighted",zero_division=0))
    for key,val in (("accuracy",acc),("macro_f1",macro),("weighted_f1",weighted)):
        if abs(original[key]-val)>1e-12:
            raise ValueError(f"Saved official {key} not reproducible from original predictions")

    # The pre-registered paper comparator uses macro_f1_reported.
    paper=cfg["paper_comparator"]
    if set(("accuracy","macro_f1_reported"))-set(paper):
        raise ValueError("Original pre-registered paper comparator missing")
    comparison={
        "our_model_metrics":original,
        "paper_aggregate_metrics":paper,
        "numeric_accuracy_difference_pp":100*(acc-paper["accuracy"]),
        "numeric_macro_f1_difference_pp":100*(macro-paper["macro_f1_reported"]),
        "comparison_status":(
            "Original official test split and five-record windowing comparable; "
            "training/preprocessing methods and paper metric averaging may differ"),
        "paper_aggregate_f1_not_assumed_weighted":True,
        "method_preselected":True,
        "official_test_evaluated_once":True,
        "recovered_from_existing_original_predictions_without_second_test":True,
        "original_script_exception":"KeyError: 'macro_f1' when reading preregistered comparator",
    }
    completed={
        "official_test_evaluated_once":True,
        "experiment_id":EXPECTED_EXPERIMENT,
        "test_csv_sha256":original["test_csv_sha256"],
        "model_checkpoint_sha256":model_sha,
        "official_prediction_sha256":prob_sha,
        "accuracy":acc,
        "macro_f1":macro,
        "num_test_windows":len(y),
        "recovered_from_existing_original_predictions_without_second_test":True,
    }
    dump(out/"paper_comparison.json",comparison)
    dump(out/"official_evaluation_complete.json",completed)
    score=audit_test(out)
    print("ORIGINAL TEST SCORES RECOVERED FROM SAVED EVIDENCE.",flush=True)
    print("UNSW official training/test CSVs: NOT OPENED during recovery.",flush=True)
    print("LightGBM training and inference: NOT REPEATED.",flush=True)
    print(f"Frozen model SHA256: {model_sha}",flush=True)
    print(f"Original probability file SHA256: {prob_sha}",flush=True)
    print(f"Test count: {len(y)} | accuracy: {acc:.6f} | "
          f"macro-F1: {macro:.6f} | weighted-F1: {weighted:.6f}",flush=True)
    print(f"SCV reported accuracy: {paper['accuracy']:.4f} | "
          f"accuracy difference (pp): {comparison['numeric_accuracy_difference_pp']:.4f}",
          flush=True)
    print(f"SCV reported macro-F1: {paper['macro_f1_reported']:.4f} | "
          f"macro-F1 difference (pp): {comparison['numeric_macro_f1_difference_pp']:.4f}",
          flush=True)
    if do_archive:
        archive_final(out,token)
        print("\nOFFICIAL-001 RECOVERED RESULTS AUDITED AND PRIVATELY ARCHIVED",flush=True)
    else:
        print("AUDIT PASS — pass --archive to upload immutable original evidence.",
              flush=True)
    return score

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--archive",action="store_true")
    args=parser.parse_args()
    recover(args.output,args.archive,os.environ.get("GITHUB_TOKEN"))
