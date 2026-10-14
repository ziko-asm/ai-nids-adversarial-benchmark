# 🛡️ AI-Powered NIDS Adversarial Robustness Benchmark

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-nids-robustness.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end research benchmark and interactive dashboard evaluating the **Adversarial Robustness** of Deep Learning-based Network Intrusion Detection Systems (NIDS) under evasion attacks (**PGD** and **FGSM**).

🌐 **Live Interactive Web App:** [ai-nids-robustness.streamlit.app](https://ai-nids-robustness.streamlit.app)

---

## 📌 Abstract & Research Overview

Deep Neural Networks (DNNs) deployed in Network Intrusion Detection Systems (NIDS) offer high detection accuracy under normal operating conditions. However, they are highly vulnerable to **adversarial evasion attacks**, where imperceptible perturbations bounded by $L_\infty$-norm constraint ($\epsilon$) mislead the classifier into categorizing malicious network traffic as benign.

This project implements an **adversarially trained Multi-Layer Perceptron (MLP)** benchmarked on network flow traffic. We evaluate empirical robustness under Projected Gradient Descent (PGD) and Fast Gradient Sign Method (FGSM) attacks, demonstrating that targeted **Adversarial Training** preserves robust detection performance under active threat vectors.

---

## 📊 Key Experimental Results

| Metric | Baseline MLP | Adversarially Trained NIDS |
| :--- | :---: | :---: |
| **Clean Accuracy (Benign Traffic)** | **99.80%** | **99.80%** |
| **Robust Accuracy (FGSM, $\epsilon=0.2$)** | ~2.10% | **83.20%** |
| **Robust Accuracy (PGD-10, $\epsilon=0.2$)** | ~0.00% (Complete Collapse) | **80.41%** |

---

## 🛠️ Key Features

1. **Interactive Research Dashboard:** Deployed live via Streamlit Cloud for real-time attack simulation.
2. **Dynamic Perturbation Analysis:** Live adjustment of perturbation magnitude ($\epsilon$) and iterative PGD steps ($1 \le k \le 20$).
3. **Custom CSV Upload:** Upload custom packet capture/flow datasets to generate adversarial perturbation samples on-the-fly.
4. **Reproducible Pipeline:** Modular PyTorch code structure for data preprocessing, adversarial training, and evaluation.

---

## 🚀 Quick Start (Local Execution)

### 1. Clone Repository & Install Dependencies
```bash
git clone [https://github.com/ziko-asm/ai-nids-adversarial-benchmark.git](https://github.com/ziko-asm/ai-nids-adversarial-benchmark.git)
cd ai-nids-adversarial-benchmark
pip install -r requirements.txt