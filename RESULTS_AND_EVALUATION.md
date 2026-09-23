# Results and Evaluation Metrics

## Integrity and scope

Metrics in this file are produced by `python src/train.py`, not copied from a proposal. The dataset is the repository's deterministic PAC-LIBERO-Lite simulator collection (900 samples / 300 base episodes), not public LIBERO demonstrations or physical-robot trials. Its restricted vocabulary makes the task separable, so these values are a ceiling effect rather than a claim of general VLA robustness.

## Protocol

The split is group-safe by base episode: 630 training, 135 validation, and 135 held-out test samples. Each base episode has referential, incidental, and conflicting text variants that stay in the same partition. Logistic Regression and Random Forest are compared for both provenance classification and action prediction. `results/model_selection.json` records validation selection. Action training uses out-of-fold provenance probabilities.

## Actual test results

| Method | Task Success | Action Macro-F1 | Priority Compliance | Legitimate-Text Utility | Over-Refusal | Role Macro-F1 | Mean inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary BC | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.067 ms/sample |
| PAC-BC | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.067 ms/sample |

Raw evidence: `results/model_comparison.csv`, `results/role_classification_report.csv`, `results/role_confusion_matrix.csv`, `results/model_selection.json`.

## Metric definitions

- **Task Success:** exact match between predicted and expert bin action.
- **Priority Compliance:** task success on conflicting-directive samples.
- **Legitimate-Text Utility:** task success on referential samples.
- **Over-Refusal:** `1 - legitimate-text utility`.
- **Role Macro-F1:** macro F1 across referential, incidental, and conflicting source roles.
- **Mean inference:** batch action-model inference time divided by test samples.




## Demo evidence

Run `python tests/test_end_to_end.py`. It proves that `models/pac_bc_pipeline.joblib` is loaded, a conflicting sign is classified as conflicting, the action model predicts left for the trusted ALPHA task, and Flask renders the result.