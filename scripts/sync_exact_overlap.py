"""Secure, restricted synchronization of aggregate exact-overlap audit artifacts."""
from pathlib import Path
import argparse
import os
import requests

REPO = "itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX = "data/manifests/unsw_nb15/exact_overlap"
FILES = ("overlap_summary.json", "per_class_overlap.csv", "overlap_report.md")


def sync(directory, token):
    if not token:
        raise RuntimeError("pushRare secret unavailable")
    contents = {}
    for name in FILES:
        path = Path(directory) / name
        if not path.is_file():
            raise FileNotFoundError(str(path))
        contents[name] = path.read_text(encoding="utf-8")
    base_url = "https://api.github.com/repos/" + REPO
    with requests.Session() as s:
        s.headers.update({
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })
        def req(method, endpoint, **kwargs):
            response = s.request(method, base_url + "/" + endpoint, timeout=90, **kwargs)
            response.raise_for_status()
            return response.json()
        head = req("GET", "git/ref/heads/main")["object"]["sha"]
        base_tree = req("GET", "git/commits/" + head)["tree"]["sha"]
        items = [{"path": PREFIX + "/" + name,
                  "mode": "100644", "type": "blob", "content": contents[name]}
                 for name in FILES]
        tree = req("POST", "git/trees", json={"base_tree": base_tree, "tree": items})["sha"]
        if tree == base_tree:
            print("Already synchronized; no changed audit artifacts.")
            return head
        new_sha = req("POST", "git/commits", json={
            "message": "data: sync SHA-256 exact-overlap audit from Kaggle",
            "tree": tree, "parents": [head]})["sha"]
        req("PATCH", "git/refs/heads/main", json={"sha": new_sha, "force": False})
    print("VERIFIED EXACT-OVERLAP SYNC COMMIT: https://github.com/" + REPO + "/commit/" + new_sha)
    return new_sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    sync(args.directory, os.environ.get("GITHUB_TOKEN"))
