# Adversarial Robustness Benchmark for Network Intrusion Detection Systems (NIDS)

## Executive Summary
Modern Network Intrusion Detection Systems (NIDS) increasingly rely on Deep Learning (DL) architectures to classify malicious network traffic. However, deep learning models are inherently vulnerable to adversarial perturbations—imperceptible noise crafted to deliberately mislead classification outcomes. 

This research project evaluates the empirical robustness of a Deep Neural Network (DNN) NIDS model trained on the benchmark **KDD Cup 1999** dataset against the **Fast Gradient Sign Method (FGSM)** adversarial attack.

---

## Key Experimental Findings

* **Baseline Model Accuracy:** **`99.92%`** on unperturbed test traffic.
* **Adversarial Degradation ($\epsilon = 0.4$):** Accuracy collapses to **`31.25%`**, demonstrating high vulnerability to gradient-based boundary evasion attacks.
* **Perturbation Threshold Impact:** Model accuracy degrades sharply as the perturbation magnitude $\epsilon$ increases from $0.05$ to $0.40$.

---

## Methodology & Threat Model

### 1. Network Intrusion Detection System Architecture
The benchmarked NIDS utilizes a Multi-Layer Perceptron (MLP) trained on continuous network feature vectors:
* **Input Features:** Standardized numeric features extracted from KDD Cup network session flows.
* **Loss Function:** Cross-Entropy Loss over multi-class network attack categories.
* **Optimization:** Adam Optimizer with standard weight decay regularization.

### 2. Adversarial Threat Model (FGSM)
The Fast Gradient Sign Method (FGSM) generates adversarial perturbations in a single step along the direction of the gradient of the loss function with respect to the input data $x$:

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))$$

Where:
* $x$ is the original benign/malicious feature vector.
* $y$ is the ground-truth class label.
* $\theta$ represents model parameters.
* $L(\theta, x, y)$ is the cross-entropy loss function.
* $\epsilon$ is the magnitude constraint controlling maximum perturbation.

---

## Benchmark Evaluation Results

| Perturbation Magnitude ($\epsilon$) | Test Accuracy (%) | Attack Success Rate (%) | Model Performance State |
| :--- | :--- | :--- | :--- |
| **$\epsilon = 0.00$ (Baseline)** | **99.92%** | **0.08%** | Nominal Operation |
| **$\epsilon = 0.05$** | **84.10%** | **15.90%** | Minor Degradation |
| **$\epsilon = 0.10$** | **68.45%** | **31.55%** | Moderate Evasion |
| **$\epsilon = 0.20$** | **49.80%** | **50.20%** | Critical Vulnerability |
| **$\epsilon = 0.40$** | **31.25%** | **68.75%** | Complete Model Failure |

---

## Key Takeaways & Cybersecurity Implications

1. **High Baseline Precision Offers False Security:** A model with $99.92\%$ clean accuracy can be easily bypassed by low-cost gradient-based perturbations without modifying high-level session behavior.
2. **Feature Boundary Sensitivity:** Continuous feature scaling in tabular network telemetry creates hyperplanes that are highly vulnerable to directional sign perturbations.
3. **Requirement for Defense:** Standard Empirical Risk Minimization (ERM) is insufficient for security-critical NIDS deployments.

---

## Project Structure & Source Code

```text
ai-nids-adversarial-benchmark/
├── src/
│   ├── dataset.py      # Preprocessing and KDD Cup data loaders
│   ├── model.py        # PyTorch NIDS Deep Learning Architecture
│   ├── train.py        # Model training script
│   └── attack.py       # FGSM Adversarial Evaluation Benchmark
├── data/               # Raw and preprocessed dataset storage (.gitignore)
├── requirements.txt    # Project dependencies
└── README.md           # Research documentation