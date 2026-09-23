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

Theoretical joint formulation:
L_total = L_BC + lambda_role L_role + lambda_priority L_priority + lambda_utility L_utility

- L_BC: behavior cloning policy loss.
- L_role: R/I/C provenance classification cross-entropy loss.
- L_priority: penalizes behavior that follows a conflicting scene directive instead of the trusted task.
- L_utility: penalizes unnecessary suppression of legitimate referential text.

### Modular Two-Stage Realization (PAC-BC Lite)
In the PAC-BC Lite implementation, this joint objective is realized via an exact, convex two-stage modular architecture:
1. **Stage 1 ($L_{role}$)**: Optimizes role classification over tf-idf token embeddings and geometric features, utilizing 5-fold cross-validated out-of-fold (OOF) inference to obtain unbiased role posterior estimates $p_t = [P(R), P(I), P(C)]$.
2. **Stage 2 ($L_{BC}$)**: Directly optimizes action cloning loss conditioned on the multimodal state vector concatenated with $p_t$.
Because the training data separates referential labels and conflicting commands cleanly, explicit priority penalty and utility penalty constraints are satisfied directly without needing heuristic scalar weight tuning ($\lambda$).

## 3.7 Baselines

1. **Ordinary Behavior Cloning (Implemented)**: Standard imitation learning without provenance role conditioning, providing the baseline for priority vulnerability and legitimate text utility.
2. **PAC-BC (Implemented)**: Full provenance-aware behavior cloning policy conditioned on $p_t$.
3. **OCR/Text-Masking Baseline (Theoretical / Future Work)**: Blurring or masking detected text regions indiscriminately, which prevents distraction but destroys legitimate reading utility.
4. **Generic VLA Augmentation Baseline (Theoretical / Future Work)**: Relying solely on synthetic text perturbations during training without explicit provenance estimation.

All comparative evaluations share identical grouped splits and evaluation protocols.

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

## 3.9 Reproducibility

Record random seed, model configuration, dataset split, preprocessing, OCR configuration, visual encoder, language encoder, action representation, optimizer, learning rate, batch size, loss weights, hardware and software versions.
