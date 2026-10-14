![CI/CD Pipeline Status](https://github.com/ziko-asm/ai-nids-adversarial-benchmark/actions/workflows/ci.yml/badge.svg)
# AI-Powered NIDS Adversarial Robustness Benchmark & Research Suite

[![Streamlit App](https://img.shields.io/badge/Streamlit-Cloud-red?style=for-the-badge&logo=streamlit)](https://ai-nids-robustness.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An advanced, production-grade research benchmark and interactive dashboard designed to evaluate, stress-test, and harden Deep Learning-based Network Intrusion Detection Systems (NIDS) against state-of-the-art adversarial perturbation attacks.

---

## 🌟 Executive Overview

Modern Network Intrusion Detection Systems powered by Deep Learning exhibit high vulnerability to adversarial machine learning. Minor, imperceptible perturbations injected into network traffic flow statistics can bypass standard neural networks entirely, causing catastrophic security failures (false negatives on malicious intrusions).

This research suite implements a rigorous benchmark comparing **Multi-Layer Perceptrons (MLP)** against **1-Dimensional Convolutional Neural Networks (1D-CNN)** under adversarial pressure, providing automated defenses, Explainable AI (XAI) attributions, and formal academic PDF reporting.

---

## 🏗️ System Architecture & Core Modules

```text
[ Network Traffic CSV ] ---> [ Data Preprocessing Pipeline ]
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
[ MLP Baseline Model ]       [ 1D-CNN Proposed Model ]    [ Adversarial Perturbation Engine ]
       |                               |                 (PGD / FGSM with Tunable ε)
       +-------------------------------+-------------------------------+
                                       |
                                       v
                     [ Interactive Streamlit Dashboard ]
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
[ Defense & Retraining ]      [ XAI Feature Attribution ]     [ Automated PDF Report ]