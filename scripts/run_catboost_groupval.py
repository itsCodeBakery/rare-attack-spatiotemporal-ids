"""CATBOOST-GROUPVAL-001: a strong single-row tabular control (validation only).

This script intentionally does NOT read or score the official testing CSV.
It uses train-only 42-feature group-disjoint validation, and records evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from rareids.baseline_utils import (canonical_group_ids, group_disjoint_stratified_split,
                                    split_indices_sha256)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def run(root, config_path, outputs, checkpoint_dir):
    from catboost import CatBoostClassifier, __version__ as cb_version
    import sklearn

    root, config_path, outputs, checkpoint_dir = map(
        Path, (root, config_path, outputs, checkpoint_dir)
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["stage"] != "training-only-validation" or not config["no_official_test_predictions"]:
        raise ValueError("Refusing a run without validation-only protocol")
    if config["device"] != "GPU":
        raise ValueError("Step 5 is configured to use Kaggle GPU only")
    manifest_path = root / config["dataset_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_entry = manifest["inputs"]["train"]
    test_entry = manifest["inputs"]["test"]
    if (train_entry["rows"], test_entry["rows"]) != (175341, 82332):
        raise ValueError("Official UNSW-NB15 partition identity has changed")
    train_path = Path(train_entry["path"])
    if not train_path.is_file():
        raise FileNotFoundError("Official training file is missing: " + str(train_path))
    if sha256(train_path) != train_entry["sha256"]:
        raise ValueError("Training SHA-256 differs from verified Step 3 manifest")
    if manifest["schema"]["common_feature_count"] != 42:
        raise ValueError("Feature count inconsistent with official data manifest")

    frame = pd.read_csv(train_path, low_memory=False)
    columns = manifest["schema"]["common_feature_names"]
    if len(frame) != 175341 or any(col not in frame for col in columns):
        raise ValueError("Official training rows/features failed integrity check")
    if frame[columns].isna().any().any():
        raise ValueError("Missing values: do not silently fit imputers")
    numeric_cols = frame[columns].select_dtypes(include="number").columns
    if not np.isfinite(frame[numeric_cols].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite numeric values found")
    if frame["attack_cat"].isna().any():
        raise ValueError("Missing attack labels")

    labels = frame["attack_cat"].astype(str).str.strip().to_numpy()
    classes = sorted(set(labels))
    if len(classes) != 10:
        raise ValueError(f"Expected 10 benchmark classes; got {classes}")
    groups = canonical_group_ids(frame, columns)
    tr_idx, va_idx, fold_id = group_disjoint_stratified_split(
        labels, groups, n_splits=config["n_splits"], seed=config["seed"]
    )
    if np.intersect1d(tr_idx, va_idx).size:
        raise AssertionError("Internal training/validation row overlap")
    X = frame[columns].copy()
    categoricals = list(X.select_dtypes(include=["object", "string", "category"]).columns)
    for col in categoricals:
        X[col] = X[col].astype(str)

    options = {k: v for k, v in config["catboost"].items() if v is not None}
    options["task_type"] = "GPU"
    options["random_seed"] = config["seed"]
    options["allow_writing_files"] = False
    options["verbose"] = 100
    model = CatBoostClassifier(**options)
    start = time.perf_counter()
    model.fit(X.iloc[tr_idx], labels[tr_idx],
              cat_features=categoricals,
              eval_set=(X.iloc[va_idx], labels[va_idx]),
              use_best_model=True)
    elapsed = round(time.perf_counter() - start, 3)
    prediction = np.asarray(model.predict(X.iloc[va_idx])).reshape(-1).astype(str)
    truth = labels[va_idx]
    p, r, f, support = precision_recall_fscore_support(
        truth, prediction, labels=classes, zero_division=0
    )
    cm = confusion_matrix(truth, prediction, labels=classes)
    normal = truth == "Normal"
    normal_fpr = float(np.mean(prediction[normal] != "Normal")) if normal.any() else None
    metrics = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "normal_to_attack_false_positive_rate": normal_fpr,
        "model_selected_best_iteration": int(model.best_iteration_),
        "elapsed_train_seconds": elapsed,
        "validation_size": int(len(va_idx)),
        "training_size": int(len(tr_idx))
    }

    outputs.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / (config["experiment_id"] + ".cbm")
    model.save_model(str(model_path))
    class_rows = [{
        "class": cls, "support": int(s), "precision": float(pi),
        "recall": float(ri), "f1": float(fi)
    } for cls, s, pi, ri, fi in zip(classes, support, p, r, f)]
    pd.DataFrame(class_rows).to_csv(outputs / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        outputs / "confusion_matrix.csv", index_label="true_class"
    )
    (outputs / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (outputs / "config_effective.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        gpu_summary = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        gpu_summary = "GPU details not reported"
    evidence = {
        "experiment_id": config["experiment_id"],
        "status": "completed_unverified",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "research_git_sha": git_head(),
        "dataset_train_sha256": train_entry["sha256"],
        "dataset_test_sha256_reference_only": test_entry["sha256"],
        "official_test_accessed_for_inference": False,
        "official_test_labels_used_for_fitting": False,
        "training_partition": "official UNSW-NB15 training, 175341 rows",
        "data_roles": "Kaggle mirror file names reversed",
        "validation": {
            "algorithm": "StratifiedGroupKFold", "number_of_folds": config["n_splits"],
            "selected_fold": fold_id, "seed": config["seed"],
            "sha256_of_train_val_indices": split_indices_sha256(tr_idx, va_idx),
            "shared_exact_feature_groups": int(len(set(groups[tr_idx]) & set(groups[va_idx]))),
            "unique_feature_groups_total": int(len(set(groups))),
            "class_counts_validation": {c: int(sum(truth == c)) for c in classes}
        },
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "catboost": cb_version
        },
        "compute": {"task_type": "GPU", "nvidia_smi": gpu_summary},
        "checkpoint": {
            "kaggle_output_path": str(model_path),
            "sha256": sha256(model_path),
            "size_bytes": model_path.stat().st_size,
            "storage_note": "Save a Kaggle notebook version to preserve model output. Checkpoint not uploaded to GitHub."
        },
        "metrics": metrics,
        "limitations": [
            "Validation subset from official training partition only, not official test",
            "CSV row order not verified chronological",
            "Official test contains known exact-feature overlap; final study must report overlap sensitivity"
        ]
    }
    (outputs / "artifact_manifest.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# CATBOOST-GROUPVAL-001 — observed validation results", "",
        "**Training-only validation; NOT a result on the official UNSW-NB15 test set.**",
        "",
        f"- Macro-F1: **{metrics['macro_f1']:.6f}**",
        f"- Weighted-F1: **{metrics['weighted_f1']:.6f}**",
        f"- Accuracy: **{metrics['accuracy']:.6f}**",
        f"- Normal-to-attack false-positive rate: **{normal_fpr:.6f}**",
        f"- Training/validation rows: **{len(tr_idx):,} / {len(va_idx):,}**",
        f"- Train/validation shared feature-signature groups: **0**",
        f"- GPU: {gpu_summary}",
        f"- Training time: {elapsed:.1f} seconds",
        f"- Git SHA: \`{evidence['research_git_sha']}\`",
        f"- Checkpoint SHA-256: \`{evidence['checkpoint']['sha256']}\`",
        "", "Every metric is calculated from saved execution in Kaggle; "
        "no official-test predictions were produced.", ""
    ]
    (outputs / "run_report.md").write_text("\n".join(report), encoding="utf-8")
    print("VALIDATION COMPLETE:", json.dumps(metrics, indent=2))
    print("Model checkpoint stored in Kaggle working directory:", model_path)
    return evidence


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoints", required=True)
    args = p.parse_args()
    run(args.repo, args.config, args.output, args.checkpoints)
