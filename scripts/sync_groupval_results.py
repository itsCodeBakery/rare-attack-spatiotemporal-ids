"""Synchronize only compact group-disjoint CatBoost validation evidence.

Kaggle secret \`pushRare\` is passed in process environment as GITHUB_TOKEN;
the token value is never printed or stored in Git.
"""
from pathlib import Path
import argparse
import os
import requests

REPO = "itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX = "results/CATBOOST-GROUPVAL-001"
FILES = ("metrics.json", "config_effective.json", "per_class_metrics.csv",
         "confusion_matrix.csv", "artifact_manifest.json", "run_report.md")


def sync(directory, token):
    if not token:
        raise RuntimeError("Kaggle pushRare secret unavailable")
    blobs = {}
    for name in FILES:
        path = Path(directory) / name
        if not path.is_file():
            raise FileNotFoundError(str(path))
        blobs[name] = path.read_text(encoding="utf-8")
    root = "https://api.github.com/repos/" + REPO
    with requests.Session() as s:
        s.headers.update({
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })
        def query(method, path, **kwargs):
            r = s.request(method, root + "/" + path, timeout=90, **kwargs)
            r.raise_for_status()
            return r.json()
        head = query("GET", "git/ref/heads/main")["object"]["sha"]
        base_tree = query("GET", "git/commits/" + head)["tree"]["sha"]
        elements = [
            {"path": PREFIX + "/" + name, "mode": "100644",
             "type": "blob", "content": data}
            for name, data in blobs.items()
        ]
        new_tree = query("POST", "git/trees",
                         json={"base_tree": base_tree, "tree": elements})["sha"]
        if new_tree == base_tree:
            print("Results already synchronized to GitHub.")
            return head
        commit = query("POST", "git/commits", json={
            "message": "results: sync group-disjoint CatBoost training-only validation",
            "tree": new_tree, "parents": [head]})["sha"]
        query("PATCH", "git/refs/heads/main", json={"sha": commit, "force": False})
    print("VERIFIED BASELINE SYNC COMMIT: https://github.com/" + REPO + "/commit/" + commit)
    return commit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    a = p.parse_args()
    sync(a.directory, os.environ.get("GITHUB_TOKEN"))
