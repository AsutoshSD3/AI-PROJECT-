# Proposed Project Architecture — PAC-BC

## 1. System Overview

PAC-BC is a provenance-aware imitation-learning framework for language-conditioned robot manipulation. It receives a trusted task instruction, an RGB observation and robot state. Visible scene text is extracted with OCR and classified as Referential (R), Incidental (I), or Conflicting Directive (C). The provenance-aware policy then predicts the robot action.

## 2. End-to-End Data Flow

    Trusted Task Instruction
              |
              v
       Instruction Encoder
              |
              | instruction embedding
              v
    +-------------------------+
    |                         |
    | RGB Camera Observation  | ---> OCR / Text Region Extractor
    |                         |              |
    +-------------------------+              v
             |                         text + bounding boxes
             |                               |
             |                               v
             |                      Provenance Classifier
             |                         R / I / C
             |                               |
             +---------------+---------------+
                             |
                             v
                  PAC-BC Behavior Policy
                             |
                             v
                    Robot Action Head
                  pose delta + gripper
                             |
                             v
                    LIBERO Environment
                             |
                             v
                 Task Success / Evaluation

## 3. Major Components

### RGB observation module
Receives camera observations from the LIBERO manipulation environment. Frames are resized and normalized before visual processing.

### Robot-state module
Provides the robot proprioception/state required for action prediction.

### OCR / text-region extractor
Detects visible text and records text string, bounding box, OCR confidence, frame ID and episode ID.

### Trusted instruction encoder
Encodes the natural-language task instruction. The trusted instruction is treated as the authoritative task objective.

### Provenance role classifier
Classifies every detected text instance into:

| Label | Meaning | Policy treatment |
|---|---|---|
| R | Legitimate task-related text | Preserve/use |
| I | Irrelevant visible text | Ignore |
| C | Untrusted conflicting directive | Reject |

The classifier outputs P(R), P(I), P(C).

### PAC-BC policy
Fuses visual features, trusted instruction representation, OCR/text representation, provenance probabilities and robot state.

The proposed training objective is:

L_total = L_BC + lambda_role L_role + lambda_priority L_priority + lambda_utility L_utility

where L_BC is behavior cloning loss, L_role is R/I/C classification loss, L_priority penalizes following conflicting scene directives, and L_utility penalizes suppressing useful referential text.

### Action head
Predicts the configured robot action, represented in the current design as end-effector pose delta plus gripper command.

### Evaluation module
Records Task Success, Priority Compliance, Legitimate-Text Utility, Over-Refusal Rate, role macro-F1, confusion matrix, action imitation error and inference latency.

## 4. Technology Stack

| Layer | Technology |
|---|---|
| Robot benchmark/simulation | LIBERO |
| ML framework | PyTorch |
| Language | Python |
| OCR | Configured OCR engine |
| Image processing | OpenCV / torchvision |
| Language representation | Transformer-based text encoder |
| Configuration | YAML |
| Data metadata | JSON / CSV |
| Version control | Git + GitHub |
| Compute | CUDA-capable NVIDIA GPU recommended |

Exact package versions and the selected OCR/text encoder must be pinned before final experiments.

## 5. Training Flow

1. Download public LIBERO demonstrations.
2. Select the target expert episodes.
3. Render RGB observations.
4. Generate R/I/C visible-text variants.
5. Store ground-truth role labels.
6. Run OCR and store detected strings and boxes.
7. Preprocess image, text, state and action data.
8. Create grouped train/validation/test manifests.
9. Train PAC-BC and baseline models.
10. Evaluate on identical test conditions.
11. Store raw logs, summaries and figures.

## 6. Review Demo

Show the same task in three conditions:

1. Referential text
2. Incidental text
3. Conflicting directive

For each case show:

Trusted instruction -> RGB frame -> OCR boxes/text -> R/I/C probabilities -> predicted action -> task result

This directly demonstrates the proposed input-to-output pipeline.
