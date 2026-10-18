"""Atomically publish only aggregate source-provenance comparison files."""
import argparse
import os
from pathlib import Path
import requests

REPO = "itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX = "data/manifests/unsw_nb15/publisher_comparison"
FILES = ("source_comparison.json", "source_comparison_report.md")


def sync(folder, token):
    if not token:
        raise ValueError("Kaggle pushRare token unavailable")
    files = {name: (Path(folder) / name).read_text(encoding="utf-8")
             for name in FILES}
    base = "https://api.github.com/repos/" + REPO
    with requests.Session() as session:
        session.headers.update({
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })
        def request(method, path, **kwargs):
            r = session.request(method, base + "/" + path, timeout=90, **kwargs)
            r.raise_for_status()
            return r.json()
        parent = request("GET", "git/ref/heads/main")["object"]["sha"]
        tree_base = request("GET", "git/commits/" + parent)["tree"]["sha"]
        tree = request("POST", "git/trees", json={
            "base_tree": tree_base,
            "tree": [{"path": PREFIX + "/" + name, "mode": "100644",
                      "type": "blob", "content": contents}
                     for name, contents in files.items()]
        })["sha"]
        if tree == tree_base:
            print("Comparison already synchronized: no changed files")
            return parent
        new_commit = request("POST", "git/commits", json={
            "message": "data: synchronize publisher-upload UNSW-NB15 source comparison",
            "tree": tree, "parents": [parent]
        })["sha"]
        request("PATCH", "git/refs/heads/main",
                json={"sha": new_commit, "force": False})
    print("VERIFIED SOURCE COMPARISON SYNC COMMIT: https://github.com/" +
          REPO + "/commit/" + new_commit)
    return new_commit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    args = p.parse_args()
    sync(args.directory, os.environ.get("GITHUB_TOKEN"))
