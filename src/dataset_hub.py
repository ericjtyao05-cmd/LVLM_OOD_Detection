"""Push / pull the experiment image dataset to a PRIVATE HuggingFace dataset repo.

Rationale: images are large binaries (and ImageNet-derived, so redistribution is
restricted) -> they live on a *private* HF dataset, not in git. Manifests (the
CSV split definitions) live in git and reference relative `data/...` paths, so a
run is fully reproducible from (pinned HF revision) x (committed manifests).

  push : upload ./data -> <repo>/data, print the commit revision (pin this).
  pull : snapshot_download <repo> -> ./ (recreates ./data) at an optional revision.

Token comes from $HF_TOKEN. Repo/visibility/revision come from experiment.yaml
(the `hf:` block) unless overridden on the CLI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config


def _token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN not set (needed for the private dataset repo).")
    return tok


def push(data_dir: str, repo_id: str, private: bool = True) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(data_dir), repo_id=repo_id, repo_type="dataset",
        path_in_repo="data", commit_message="dataset snapshot",
        ignore_patterns=["*_cache*", "**/_*"])
    sha = getattr(info, "oid", None)
    if not sha:
        sha = api.list_repo_commits(repo_id, repo_type="dataset")[0].commit_id
    print(f"[hf] pushed {data_dir} -> {repo_id} (private={private})")
    print(f"[hf] revision: {sha}   <- pin this in experiment.yaml hf.revision")
    return sha


def pull(repo_id: str, dest: str = ".", revision: str | None = None) -> str:
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        token=_token(), local_dir=str(dest))
    print(f"[hf] pulled {repo_id}@{revision or 'latest'} -> {Path(dest)/'data'}")
    return path


def _hf_cfg(cfg: dict) -> dict:
    hf = cfg.get("hf", {})
    if not hf.get("dataset_repo"):
        raise SystemExit("experiment.yaml is missing hf.dataset_repo")
    return hf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dest", default=".")
    ap.add_argument("--repo", default=None, help="override hf.dataset_repo")
    ap.add_argument("--revision", default=None, help="override hf.revision (pull)")
    args = ap.parse_args()

    hf = _hf_cfg(load_config(args.config))
    repo = args.repo or hf["dataset_repo"]
    if args.action == "push":
        push(args.data_dir, repo, private=hf.get("private", True))
    else:
        pull(repo, args.dest, revision=args.revision or hf.get("revision"))


if __name__ == "__main__":
    main()
