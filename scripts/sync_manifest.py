"""Publish only four aggregate Kaggle audit files to GitHub in one atomic commit.
Requires a fine-grained GITHUB_TOKEN secret with Contents read/write for this repo only.
"""
import argparse, os
from pathlib import Path
import requests

NAMES = ("audit_summary.json", "column_schema.csv", "class_distribution.csv", "audit_report.md")
PREFIX = "data/manifests/unsw_nb15"

def publish(folder, token, repo="itsCodeBakery/rare-attack-spatiotemporal-ids", branch="main"):
    if not token:
        raise ValueError("No GITHUB_TOKEN available; not synchronized")
    files = {name: (Path(folder)/name).read_text(encoding="utf-8") for name in NAMES}
    root = f"https://api.github.com/repos/{repo}"
    with requests.Session() as s:
        s.headers.update({"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"})
        def req(method, endpoint, **kwargs):
            resp = s.request(method, f"{root}/{endpoint}", timeout=60, **kwargs)
            resp.raise_for_status()
            return resp.json()
        head = req("GET", f"git/ref/heads/{branch}")["object"]["sha"]
        base_tree = req("GET", f"git/commits/{head}")["tree"]["sha"]
        items = [{"path": f"{PREFIX}/{name}", "mode": "100644",
                  "type": "blob", "content": contents} for name,contents in files.items()]
        tree = req("POST", "git/trees", json={"base_tree": base_tree, "tree": items})["sha"]
        if tree == base_tree:
            print("Already synchronized (no changed manifest files).")
            return head
        sha = req("POST", "git/commits", json={"message": "data: register Kaggle UNSW-NB15 audit",
                   "tree": tree, "parents": [head]})["sha"]
        req("PATCH", f"git/refs/heads/{branch}", json={"sha": sha, "force": False})
    print(f"VERIFIED SYNC COMMIT: https://github.com/{repo}/commit/{sha}")
    return sha

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    a = p.parse_args()
    publish(a.directory, os.environ.get("GITHUB_TOKEN", ""))
