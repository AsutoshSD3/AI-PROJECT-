# Chapter 3 — Proposed Methodology

## 3.1 Overview

PAC-BC is a provenance-aware imitation-learning framework for language-conditioned robot manipulation. The system distinguishes task-relevant visible text from irrelevant or conflicting scene text while preserving useful label-reading capability.

The trusted task instruction is treated as the primary objective.

## 3.2 Architecture

The system contains:

1. RGB observation and robot-state acquisition
2. OCR/text-region extraction
3. trusted instruction encoding
4. R/I/C provenance classification
5. PAC-BC multimodal policy fusion
6. robot action prediction
7. environment evaluation

The complete architecture is documented in docs/ARCHITECTURE.md.

## 3.3 Provenance Classes

R — Referential: legitimate task-related text.

I — Incidental: visible text unrelated to the task.

C — Conflicting Directive: untrusted text that contradicts the trusted task.

## 3.4 Multimodal Representation

Let I_t be the RGB observation, s_t the robot state, q the trusted instruction and T_t the OCR text regions.

Visual representation:
v_t = E_v(I_t)

State representation:
h_t = E_s(s_t)

Instruction representation:
l = E_q(q)

OCR representation:
o_t = E_o(T_t)

Role probabilities:
p_t = softmax(E_r(v_t, l, o_t))

where p_t = [P(R), P(I), P(C)].

## 3.5 PAC-BC Policy

The policy fuses:

- visual representation
- robot state
- trusted instruction
- OCR representation
- provenance probabilities

The fused representation is:

z_t = Fuse(v_t, h_t, l, o_t, p_t)

The action head predicts:

a_hat_t = pi(z_t)

The configured action contains robot pose delta and gripper command.

## 3.6 Training Objective

L_total = L_BC + lambda_role L_role + lambda_priority L_priority + lambda_utility L_utility

L_BC: behavior cloning loss.

L_role: R/I/C classification loss.

L_priority: penalizes behavior that follows a conflicting scene directive instead of the trusted task.

L_utility: penalizes unnecessary suppression of legitimate referential text.

The final experiment must report the selected loss weights.

## 3.7 Baselines

1. Ordinary behavior cloning
2. OCR/text-masking baseline
3. Generic visual-language augmentation baseline
4. PAC-BC

All models use identical grouped splits and evaluation conditions.

## 3.8 Inference

Trusted instruction
-> RGB + robot state
-> OCR
-> encode image/text/instruction/state
-> predict R/I/C probabilities
-> provenance-aware fusion
-> action prediction
-> environment execution
-> task evaluation

## 3.9 Review Demonstration

Use three controlled cases from the same task/seed:

- Referential
- Incidental
- Conflicting

For each case display trusted instruction, RGB frame, OCR result, R/I/C probabilities, predicted action and final task result.

## 3.10 Reproducibility

Record random seed, model configuration, dataset split, preprocessing, OCR configuration, visual encoder, language encoder, action representation, optimizer, learning rate, batch size, loss weights, hardware and software versions.
