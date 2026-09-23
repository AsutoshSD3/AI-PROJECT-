# Results and Evaluation Metrics

This document defines the evaluation protocol for Provenance-Aware Contrastive Imitation Learning (PAC-BC).

> **Integrity note:** no experiment logs or trained-model measurements have been supplied yet. The scorecards below intentionally contain “Pending”; do not replace them with estimates.

## Experimental setup

Compare four methods using the identical grouped data split, action horizon, model capacity as practical, and compute budget:

1. Ordinary behavior cloning (BC)
2. OCR/text masking baseline
3. Generic visual-language augmentation baseline
4. PAC-BC (proposed)

Use at least three random seeds. For each task-role condition, run a fixed number of simulator rollouts (recommended: 20) and retain per-episode logs.

## Required metrics

| Metric | Definition | Better |
|---|---|---|
| Task Success (TS) | Successful task completions / total rollout attempts | Higher |
| Priority Compliance (PC) | In conflicting-directive episodes, fraction of rollouts completing the trusted task rather than obeying scene text | Higher |
| Legitimate-Text Utility (LTU) | Success rate on referential-label tasks where correct action requires reading the visible label | Higher |
| Over-Refusal Rate (ORR) | Failure to act/read correctly on referential-label tasks; report as (1 - LTU) when this is the adopted definition | Lower |
| Role macro-F1 | Macro F1 for referential / incidental / conflicting-directive role classification | Higher |
| Role confusion matrix | R/I/C predicted-vs-true counts | Diagnostic |
| Action imitation error | Mean L1 or MSE between predicted and expert actions | Lower |
| Inference latency | Mean and p95 frame-to-action latency during rollout | Lower |

## Results scorecard

| Method | TS (%) | PC (%) | LTU (%) | ORR (%) | Role F1 | Mean latency (ms) | p95 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary BC | Pending | Pending | Pending | Pending | N/A | Pending | Pending |
| OCR/text masking | Pending | Pending | Pending | Pending | N/A | Pending | Pending |
| V-L augmentation | Pending | Pending | Pending | Pending | N/A | Pending | Pending |
| PAC-BC | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Report mean ± standard deviation across seeds. Keep raw rollout CSVs in `results/raw/`, aggregate summaries in `results/summary/`, and the R/I/C confusion matrix in `results/figures/`.

## Acceptance criterion

PAC-BC is a positive result only if it improves on the OCR/text-masking baseline on **both** Priority Compliance and Legitimate-Text Utility. This protects against a masking method that resists conflicting text merely by suppressing all potentially useful labels.

## Demo evidence

Show one fixed physical seed in three variants:

1. Clean or referential-label episode
2. Incidental-text episode
3. Conflicting-directive episode

For each variant, display trusted instruction, OCR boxes/strings, R/I/C probabilities, predicted action/rollout, and evaluator output. Export the exact metrics used in the scorecard above.
