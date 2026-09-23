# AI PROJECT — HANDOFF

## Current Situation

This project was previously worked on using OpenAI Codex.

Codex reached its usage limit before the work was completely finished.

The existing work has been preserved in Git on branch:

`codex-backup-2026-09-23`

This branch has also been pushed to GitHub.

## IMPORTANT

Do NOT start the project from scratch.

Do NOT delete or replace the existing implementation.

Do NOT reset the repository.

Do NOT make large architectural changes without first understanding the current implementation.

The goal is to continue from the existing state.

## Files Added/Modified by the Previous Agent

### Application

- `app/app.py`
- `app/static/style.css`
- `app/templates/index.html`

### ML / Project Pipeline

- `src/__init__.py`
- `src/pipeline.py`
- `src/train.py`
- `src/analyze.py`
- `src/evaluate.py`

### Tests

- `tests/test_end_to_end.py`

### Configuration

- `requirements.txt`
- `.gitignore`

### Documentation

- `README.md`
- `RESULTS_AND_EVALUATION.md`

### Data / Model / Results

- `data/raw/pac_libero_lite.csv`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `models/pac_bc_pipeline.joblib`
- files under `results/`

## First Task

Before changing any code:

1. Inspect the complete repository structure.
2. Read `README.md`.
3. Read `RESULTS_AND_EVALUATION.md`.
4. Inspect all files under `src/`.
5. Inspect all files under `app/`.
6. Inspect `tests/test_end_to_end.py`.
7. Inspect the generated results and model files.
8. Inspect Git history.

Then give a report containing:

### Completed
What is already implemented and working.

### Partially Completed
What exists but needs improvement or verification.

### Missing
What is not yet implemented.

### Problems
Any errors, inconsistencies, broken imports, missing dependencies, incorrect documentation, or questionable implementation.

### Project Review Readiness
What still needs to be done before faculty/project review.

## Development Priority

Prioritize:

1. Correctness
2. Completing required project-review items
3. Testing
4. Documentation
5. Clean project structure
6. Faculty-review readiness

Do NOT prioritize unnecessary new features.

## Git Safety

Before making major modifications:

```bash
git status
git log --oneline -10
