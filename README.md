# Provenance-Aware Contrastive Imitation Learning (PAC-BC)

DA2 project for Vision-Language Robot Policies.

## Project goal

PAC-BC teaches a language-conditioned robot policy to distinguish visible text in a camera frame by provenance:

- **Referential**: a legitimate label that the trusted task requires the robot to read.
- **Incidental**: irrelevant scene text that should not affect behavior.
- **Conflicting directive**: untrusted scene text that contradicts the trusted task instruction.

The policy must follow the trusted instruction while preserving useful label-reading behavior.

## Architecture

```text
RGB camera + robot state ──> OCR/text-region extractor ──┐
                                                         ├─> source-role features (R/I/C)
Trusted task instruction ──> instruction encoder ───────┘
                                                               │
                                                               v
                                              PAC-BC behavior-cloning policy
                                                               │
                                                               v
                                            robot action: pose delta + gripper
```

Training combines behavior cloning with role classification, priority-consistency, and utility-preservation losses.

## Dataset

The public base benchmark is [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), which supplies language-conditioned manipulation tasks, RGB observations, proprioception, PDDL scene descriptions, and human-teleoperated demonstrations.

For the DA2 pilot, **PAC-LIBERO** is a derived paired-episode dataset:

- 300 base expert episodes (target)
- 3 visible-text roles per base episode: referential, incidental, conflicting directive
- 900 target rendered variants
- Grouped train/validation/test split: 70% / 15% / 15%, split by base episode template and text-string family to prevent leakage

See [RESULTS_AND_EVALUATION.md](RESULTS_AND_EVALUATION.md) for the full experimental protocol and scorecard.

## Repository structure

```text
README.md
RESULTS_AND_EVALUATION.md
docs/                         # final report and figures
src/                          # data, OCR, policy, training, evaluation modules
configs/                      # reproducible experiment configurations
results/                      # raw metrics and plots after experiments
```

## References

1. B. Liu et al., “LIBERO: Benchmarking Knowledge Transfer in Lifelong Robot Learning,” NeurIPS Datasets and Benchmarks, 2023.
2. LIBERO source code and datasets: https://github.com/Lifelong-Robot-Learning/LIBERO

## Current status

The proposed methodology and evaluation plan are documented. Measured results are **not yet available**; no numerical performance claims are made until simulator experiments are executed and logged.
