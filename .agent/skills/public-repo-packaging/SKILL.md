---
name: public-repo-packaging
description: Prepares safe public repository commits by enforcing allowlist-only staging and denylist checks. Use before publishing a repository or before any commit to public remote.
---

# Public Repo Packaging

## Use this skill when
- Preparing a public repository for first publish
- Creating commits for a public remote
- Verifying that no data or secrets are staged

## Checklist
1. Run `git status` and review all changed and untracked files.
2. Validate staged and unstaged files against denylist.
3. Stage only allowlisted files.
4. Confirm final staged file list before commit.

## Allowlist
- `*.ipynb`
- `evaluate_model.py`
- `README.md`
- `*evaluation*.md`
- `*vyhodnoceni*.md`

## Denylist
- `*.csv`, `*.zip`, `*.gz`, `*.7z`
- `*.parquet`, `*.feather`
- `*.pkl`, `*.joblib`, `*.onnx`, `*.pt`, `*.h5`
- `*.db`, `*.sqlite`
- `.env`, `*.key`, `*.pem`, `*.secrets*`
- `data/`, `datasets/`, `raw/`, `exports/`, `venv/`, `__pycache__/`

## Enforcement
- If any denylisted file is staged, unstage it immediately.
- Stop commit flow and report removed files.
- Continue only when staged set contains allowlisted files only.
