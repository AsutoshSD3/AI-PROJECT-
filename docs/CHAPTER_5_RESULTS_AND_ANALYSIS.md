# Chapter 5 — Results and Analysis

## 5.1 Experimental Setup & Evaluation Protocol

The empirical evaluation of PAC-BC Lite benchmarks provenance-aware behavior cloning against standard behavior cloning across the deterministic PAC-LIBERO-Lite simulator suite.

### Evaluation Metrics
1. **Task Success Rate**: Proportion of test episodes where the predicted manipulation action matches the ground-truth optimal action.
2. **Action Macro-F1**: Unweighted mean of F1 scores across target action classes (`left`, `center`, `right`).
3. **Priority Compliance**: Success rate specifically evaluated on episodes containing Conflicting Directives (`role == 'conflicting'`), assessing whether the policy adheres to trusted instructions over untrusted scene distractions.
4. **Legitimate Text Utility**: Success rate on episodes where the visible text provides essential grounding information (`role == 'referential'`), verifying that the model retains reading utility.
5. **Over-Refusal Rate**: Proportion of referential episodes improperly dismissed ($1 - \text{Legitimate Text Utility}$).
6. **Role Macro-F1**: Macro-averaged F1 score of the R/I/C provenance classifier across Referential, Incidental, and Conflicting classes.
7. **Inference Latency**: Average per-step wall-clock decision time in milliseconds.

### Data Splits
- **Train**: 630 episodes (210 base episodes)
- **Validation**: 135 episodes (45 base episodes)
- **Test**: 135 episodes (45 base episodes)
- Strict `GroupShuffleSplit` on `base_id` ensures zero episode leakage across splits.

---

## 5.2 Model Selection and Validation Results

Candidate classifiers (Logistic Regression with TF-IDF and Random Forest) were evaluated on the held-out validation split.

| Model Stage | Candidate Architecture | Validation Macro-F1 | Selection Outcome |
|---|---|---:|---|
| **Role Classifier** | Logistic Regression (C=1.0) | 1.0000 | **Selected** (Lower latency, calibrated logits) |
| | Random Forest (n=250) | 1.0000 | Candidate |
| **PAC-BC Action Policy** | Logistic Regression (C=1.0) | 1.0000 | **Selected** (Optimal linear boundary on fused features) |
| | Random Forest (n=250) | 1.0000 | Candidate |
| **Baseline Action Policy** | Logistic Regression (C=1.0) | 1.0000 | **Selected** (Identical capacity for fair comparison) |
| | Random Forest (n=250) | 1.0000 | Candidate |

Logistic Regression was chosen for deployment due to sub-millisecond inference latency (0.06 ms/sample) and well-calibrated posterior probability outputs.

---

## 5.3 Comparative Performance

The primary test-set performance comparison between Ordinary Behavior Cloning (naive baseline) and Provenance-Aware Behavior Cloning (PAC-BC) is reported below:

| Method | Task Success | Macro-F1 | Priority Compliance | Legitimate Text Utility | Over-Refusal Rate | Role Macro-F1 | Mean Inference (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ordinary BC** | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | N/A | ~0.062 ms |
| **PAC-BC** | **1.000** | **1.000** | **1.000** | **1.000** | **0.000** | **1.000** | ~0.062 ms |

### Key Findings
1. **Separability in PAC-LIBERO-Lite**: On the controlled simulator benchmark, the vocabulary tokens of trusted instructions and conflicting distractor prefixes ("IGNORE THE TASK") are linearly separable.
2. **Explicit Provenance Guarantees**: While both models achieved top task performance under deterministic inputs, PAC-BC explicitly generates auditable, calibrated posterior probabilities ($P(R), P(I), P(C)$). This provides verifiable explainability for safety-critical robotic deployment.
3. **Out-of-Fold (OOF) Calibration**: In PAC-BC, the action policy is conditioned on out-of-fold role probabilities during training, preventing over-optimistic feature reliance and aligning training feature distributions with test-time inference.

---

## 5.4 Provenance Classification Analysis

The role classification head identifies whether visible scene text is Referential ($R$), Incidental ($I$), or Conflicting ($C$).

### Classification Report (Test Set, N=135)
| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| **Conflicting** | 1.000 | 1.000 | 1.000 | 45 |
| **Incidental** | 1.000 | 1.000 | 1.000 | 45 |
| **Referential** | 1.000 | 1.000 | 1.000 | 45 |
| **Macro Average** | **1.000** | **1.000** | **1.000** | **135** |

### Confusion Matrix
```
Predicted ->   Conflicting  Incidental  Referential
Actual
Conflicting        45           0            0
Incidental          0          45            0
Referential         0           0           45
```
Zero misclassifications occurred between legitimate referential task text and adversarial conflicting prompts.

---

## 5.5 Feature Attribution and Sensitivity Analysis

Feature importance analysis extracted from the trained action model weights reveals the primary drivers of policy action selection:

| Rank | Feature Token / Representation | Coefficient Weight | Attribution Rationale |
|---:|---|---:|---|
| 1 | `text__charlie` | 1.726 | Referential target bin label grounding |
| 2 | `text__bravo` | 1.720 | Referential target bin label grounding |
| 3 | `text__alpha` | 1.702 | Referential target bin label grounding |
| 4 | `text__charlie scene` | 1.505 | Co-occurrence of scene context with target |
| 5 | `text__labeled charlie` | 1.505 | Syntactic instruction pattern matching |
| 6 | `text__labeled bravo` | 1.498 | Syntactic instruction pattern matching |
| 7 | `text__bravo scene` | 1.498 | Co-occurrence of scene context with target |
| 8 | `text__alpha scene` | 1.478 | Co-occurrence of scene context with target |
| 9 | `text__labeled alpha` | 1.478 | Syntactic instruction pattern matching |
| 10 | `text__box in` / `text__pink` | 0.765 | Manipulated object identifier grounding |

The feature weights confirm that action predictions are firmly anchored on the target object identity and destination bin labels specified in the trusted instruction, rather than spurious noise or distractor phrases.

---

## 5.6 Runtime Latency and Efficiency

Real-time robotic control requires low-latency decision cycles:
- **Feature Extraction & Tokenization**: ~0.025 ms
- **Role Estimation ($E_r$)**: ~0.020 ms
- **Policy Prediction ($\pi$)**: ~0.017 ms
- **Total Pipeline Latency**: **~0.062 ms** (over 16,000 Hz throughput on CPU)
- **Model Footprint**: 28 KB saved model artifact (`pac_bc_pipeline.joblib`), well suited for lightweight embedded robot onboard controllers.
