# ADAM: Autonomous Deterrent and Alert Module

A real-time threat detection system built on Raspberry Pi 5, combining a fine-tuned YOLOv5 model with LIME-based interpretability and FGSM adversarial robustness evaluation. This repository contains the ML components of the ADAM system: model fine-tuning, interpretability analysis, and robustness testing.

> **Note:** The full ADAM system includes hardware components (enclosure, servo, LEDs, speaker, web interface) built by Nathan Kopacz. This repo focuses on the ML pipeline built by Harshit Singh and August Garibay.

---

## Overview

Traditional security systems rely on motion sensors and require constant human oversight. ADAM replaces passive detection with an active ML-powered classifier that distinguishes threats from non-threats in real time on constrained edge hardware.

The ML pipeline fine-tunes YOLOv5n on a custom binary threat dataset, evaluates robustness under Gaussian noise and FGSM adversarial attacks, and uses LIME to diagnose model behavior and background bias.

**Key result:** Adversarial training raised clean accuracy from **69.15% to 72.63%**, deployed to Raspberry Pi 5 with a live camera feed for real-time edge inference.

---

## System Architecture

**Hardware (Nathan Kopacz):** Raspberry Pi 5, camera, LEDs, servo, speaker, 3D-printed enclosure, web interface for manual deterrent activation and live video streaming.

**ML pipeline (Harshit Singh and August Garibay):**
- Fine-tuned YOLOv5n6 on a custom binary threat dataset
- LIME interpretability to diagnose pixel-level decision making and background bias
- FGSM adversarial attack evaluation across noise and epsilon combinations
- Adversarial training to improve robustness

The model maps input images to a binary prediction:

```
C(x) = Threat  if sigmoid(W · Y(x)) >= 0.5
        Safe    otherwise
```

Where `Y(x)` is the YOLOv5 backbone output and `W` is a fine-tuned linear projection layer.

---

## Dataset

A custom binary classification dataset was constructed from two public sources:

| Source | Role | Categories Used |
|---|---|---|
| HOD Benchmark Dataset | Positive (threat) examples | Insulting Gesture, Gun, Knife |
| Human Segmentation Dataset | Negative (safe) examples | General human activity images |

Remaining HOD categories (Alcohol, Blood, Cigarette) were treated as non-threats and merged into the safe set.

| Split | Images |
|---|---|
| Training | 8,736 |
| Testing | 2,185 |

Both splits are class-balanced. Images vary in lighting, angle, and distance to improve generalization.

---

## Model Training

YOLOv5n6 was fine-tuned via transfer learning starting from pretrained weights. The final convolutional layer was replaced with a linear projection head for binary classification.

**Loss function:** Modified binary cross-entropy with an L2 penalty to prevent divergence from pretrained weights:

```
argmin  -L(x) log t  +  λ Σ |θᵢ - θ'ᵢ|²
```

Where `λ = 0.01`, batch size = 50, optimizer = SGD with early stopping (patience = 7).

---

## Interpretability with LIME

LIME (Local Interpretable Model-Agnostic Explanations) was used to identify which image regions drive predictions across three conditions:

- Pretrained YOLO without fine-tuning
- Fine-tuned YoloThreat model (clean input)
- Fine-tuned YoloThreat model (FGSM adversarial input)

**Key findings:**
- The pretrained model attends to semantically meaningful regions (edges, human shapes), as expected from ImageNet representations
- The fine-tuned model shows a strong preference for threat objects (guns) in correctly classified images, but background pixel bias was detected in misclassified examples, indicating a training data artifact
- FGSM adversarial inputs produce erratic, unstructured LIME masks with no correlation to image content, visually confirming how the attack disrupts model attention

Faithfulness scores (difference between original and masked prediction) are reported per image to validate explanation quality.

---

## Robustness Testing

### Noise Resilience

Gaussian noise was applied at levels 0.0 to 0.5. Accuracy of the baseline model drops from 69.15% at noise level 0 to around 55% at noise level 0.5. The adversarial model holds better, dropping from 72.63% to 59.03% across the same range, with recall increasing at higher noise levels.

| Model | Accuracy (noise=0) | Accuracy (noise=0.5) |
|---|---|---|
| Baseline | 69.15% | 55.15% |
| Adversarial | 72.63% | 59.04% |

### FGSM Adversarial Attacks

FGSM attacks were evaluated at epsilon values 0.0 to 0.5, both alone and combined with Gaussian noise.

| Condition | Accuracy (epsilon=0.1) | F1 (epsilon=0.1) |
|---|---|---|
| Baseline + FGSM | 43.38% | 0.227 |
| Adversarial + FGSM | 5.58% | 0.061 |

The naively trained adversarial model did not provide defensive utility against FGSM. When noise and FGSM are combined, all metrics collapse to 0% beyond noise level 0.25. At large epsilon values, both models converge to ~50% accuracy, consistent with random guessing behavior caused by large gradient steps pushing predictions into the entropy regime.

---

## Key Takeaways

- Adversarial training with simple dataset augmentation (adding FGSM examples) is not the conventional approach and does not provide reliable defense; adversarial examples should be incorporated into the training loss itself for meaningful robustness gains.
- LIME effectively revealed background pixel bias in the fine-tuned model, a finding that would not be visible from accuracy metrics alone.
- The combination of FGSM and Gaussian noise is far more damaging than either perturbation alone, collapsing all metrics to zero for both model variants at moderate noise levels.
- For a real deployment, recall is the more critical metric: false negatives remove system utility entirely, while false positives create liability.

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision ultralytics lime numpy matplotlib scikit-learn
```

### 2. Clone the repo

```bash
git clone https://github.com/harshitsingh7848/ADAM.git
cd ADAM
```

### 3. Download datasets

Place the HOD Benchmark Dataset and Human Segmentation Dataset under `data/` before running training notebooks.

---

## Repository Structure

```
.
├── notebooks/         # Fine-tuning, LIME analysis, robustness evaluation
├── src/               # Model definition, dataset loading, FGSM attack utilities
├── models/            # Saved YoloThreat weights
├── data/              # Training and test splits (download separately)
└── figures/           # Noise and epsilon performance plots, LIME mask visualizations
```

---

## Contributions

**Harshit Singh and August Garibay (ML):**
- Fine-tuned YOLOv5n6 on the custom binary threat dataset with L2-regularized transfer learning
- Diagnosed background-pixel bias via LIME analysis across pretrained, fine-tuned, and adversarial input conditions
- Evaluated model robustness under Gaussian noise (levels 0.0 to 0.5) and FGSM attacks (epsilon 0.0 to 0.5), including combined conditions
- Applied adversarial training, raising clean accuracy from 69.15% to 72.63%
- Deployed the compressed model to Raspberry Pi 5 with live camera inference

**Nathan Kopacz (Hardware):**
- Hardware enclosure, servo control, speaker integration, LEDs, and web interface

---

## Authors

**Harshit Singh**, Colorado State University, harshit.singh@colostate.edu

**August Garibay**, Colorado State University, agaribay@colostate.edu

**Nathan Kopacz**, Colorado State University, ndkopacz@gmail.com

---

## References

- Goodfellow et al., "Explaining and Harnessing Adversarial Examples," arXiv:1412.6572, 2014.
- Ribeiro et al., "Why Should I Trust You? Explaining the Predictions of Any Classifier," KDD 2016.
- HOD Benchmark Dataset: https://github.com/poori-nuna/HOD-Benchmark-Dataset
- Human Segmentation Dataset: https://github.com/VikramShenoy97/Human-Segmentation-Dataset
