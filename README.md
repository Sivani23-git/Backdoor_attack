# Selective Backdoor Defense & Trigger-Agnostic Mitigation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end framework for investigating, detecting, mitigating, and explaining backdoor attacks in Transformer-based NLP models (specifically BERT on sentiment classification benchmarks such as SST-2). 

This repository implements both **selective known-trigger defense strategies** and an **active trigger-agnostic discovery framework** that detects unseen backdoor tokens without requiring a predefined trigger dictionary.

---

## 📌 Table of Contents
- [Core Motivations & Architecture](#-core-motivations--architecture)
- [Key Features](#-key-features)
- [Empirical Performance & Benchmark Results](#-empirical-performance--benchmark-results)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage & Workflow](#-usage--workflow)
  - [1. Configuration & Data Validation](#1-configuration--data-validation)
  - [2. Model Training](#2-model-training)
  - [3. Defense Evaluation](#3-defense-evaluation)
  - [4. Explainability & Attribution (LIME / SHAP)](#4-explainability--attribution-lime--shap)
  - [5. Trigger-Agnostic & Unknown-Trigger Experiments](#5-trigger-agnostic--unknown-trigger-experiments)
  - [6. Interactive Demonstration](#6-interactive-demonstration)
- [Running Tests](#-running-tests)
- [Citation & Reference](#-citation--reference)

---

## 🎯 Core Motivations & Architecture

Backdoor attacks in NLP stealthily inject trigger tokens (e.g., `bb`, `cf`, `mn`, `tq`, `mb`) into training texts to force target predictions at test time while preserving high clean accuracy. Traditional defenses often require prior knowledge of the trigger set. 

This project tackles backdoor mitigation from two complementary angles:

1. **Selective Defense (`REMOVE` / `MASK`)**:
   - Applies targeted text sanitation (token ablation or token substitution) on detected triggers.
   - Preserves benign context while neutralizing adversarial backdoor mechanisms.

2. **Trigger-Agnostic Behavioral Discovery**:
   - Operates **without any hardcoded trigger dictionary**.
   - Identifies adversarial tokens through **deterministic single-token ablation** against a frozen model, measuring:
     - Target class probability deltas ($\Delta P_{\text{target}}$)
     - Prediction flips ($\mathbb{I}_{\text{flip}}$)
     - Confidence drops ($\Delta \text{Conf}$)
   - Cross-verifies candidates via **Multi-Context Consistency Scoring** before applying a 3-way confidence decision (*Confirmed*, *Uncertain*, *No Trigger*).

```
Input Text
    │
    ▼
1. Token Candidate Generation (Arbitrary token spans & character offsets)
    │
    ▼
2. Behavioral Influence Analysis (Single-token ablation against frozen BERT)
    │
    ▼
3. Multi-Context Validation (Empirical consistency tracking across contexts)
    │
    ▼
4. Candidate Scoring & 3-Way Confidence Gating
    ├─ CONFIRMED_TRIGGER_CANDIDATE  ──► REMOVE / MASK Cleaning ──► Frozen BERT
    ├─ UNCERTAIN_CANDIDATE          ──► Abstain (Text Unchanged) ──► Frozen BERT
    └─ NO_TRIGGER_CANDIDATE         ──► Abstain (Text Unchanged) ──► Frozen BERT
                                                                          │
                                                                          ▼
                                                                SHAP / LIME Attribution
                                                                          │
                                                                          ▼
                                                                   Final Prediction
```

---

## 🚀 Key Features

- **Backdoor Attack Benchmark**: Standardized BadNets implementation on the Stanford Sentiment Treebank (SST-2) with verified insertion triggers (`bb`, `cf`, `mn`, `tq`, `mb`).
- **Comprehensive Defenses**:
  - `UNDEFINED`: Baseline unmitigated inference on poisoned models.
  - `REMOVE`: Strips identified trigger tokens from inputs.
  - `MASK`: Replaces identified trigger tokens with a neutral mask token (`[MASK]` or `[TRIGGER]`).
- **Explainability & Model Interpretability**:
  - Integration with **LIME** and **SHAP** to quantitatively measure token attribution shares before and after defense.
  - Proves empirical attribution suppression on adversarial tokens.
- **Trigger-Agnostic Engine**:
  - Candidate token generator and ablation-based behavioral influence scorer.
  - Multi-context validator to filter out false positives on benign sentiment words (*terrible*, *great*, *not*).
  - Leave-One-Trigger-Out (LOTO) cross-validation and threshold sensitivity evaluation.
- **Production-Ready & Fully Tested**:
  - Unit and integration test suite across all defense components, training loops, and explainability pipelines.

---

## 📊 Empirical Performance & Benchmark Results

Evaluated on BERT (`bert-base-uncased`) fine-tuned on the BadNet-poisoned SST-2 dataset:

### Defense Performance on SST-2

| Defense Condition | Clean Accuracy (CACC) | Clean Accuracy Drop (CAD) | Attack Success Rate (ASR) | Absolute ASR Reduction | Relative ASR Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNDEFINED** (Baseline) | **91.17%** | 0.00 pp | 95.29% | 0.00 pp | 0.00% |
| **REMOVE** | **91.17%** | **0.00 pp** | **6.47%** | **88.82 pp** | **93.21%** |
| **MASK** | **91.17%** | **0.00 pp** | **7.57%** | **87.72 pp** | **92.06%** |

*Key Findings:*
- **Zero Clean Accuracy Drop**: Both `REMOVE` and `MASK` maintain 100% of clean test-set performance (91.17% CACC, 0.00 pp drop).
- **Massive ASR Reduction**: Attack Success Rate drops from **95.29%** down to **6.47%** (`REMOVE`) and **7.57%** (`MASK`).
- **Attribution Neutralization**: In baseline poisoned models, LIME/SHAP rank trigger tokens as the #1 most influential feature. Post-defense, trigger attribution drops to negligible or negative values.

---

## 📁 Repository Structure

```
Selective-Backdoor-Defense/
├── configs/                          # Experiment configuration files
│   ├── __init__.py
│   ├── config_loader.py              # Configuration loader & validator
│   └── default_config.yaml           # Primary experiment & model hyperparameters
├── data/                             # Raw & preprocessed benchmark datasets
│   └── raw/onion/
│       ├── badnets/sst-2/            # BadNet poisoned SST-2 splits (train, dev, test)
│       └── clean_data/sst-2/         # Clean reference SST-2 splits
├── dataset/                          # PyTorch Dataset loaders & schema definitions
│   ├── onion_loader.py
│   └── schema.py
├── defenses/                         # Core defense modules
│   ├── text_cleaner.py               # REMOVE & MASK text cleaning
│   └── trigger_detector.py           # Trigger detection utilities
├── experimental/                     # Advanced & Trigger-Agnostic research
│   ├── trigger_agnostic/             # Full vocabulary-free defense pipeline
│   │   ├── candidate_generator.py    # Span & token candidate generation
│   │   ├── defense_pipeline.py       # End-to-end trigger-agnostic pipeline
│   │   ├── evaluation.py             # Evaluation routines
│   │   ├── explainability.py         # Attribution wrappers
│   │   ├── multicontext_validator.py # Multi-context consistency verification
│   │   ├── schemas.py                # Data models
│   │   ├── token_influence.py        # Ablation & behavioral influence analysis
│   │   └── trigger_scorer.py         # Confidence gating & candidate scoring
│   └── unknown_trigger/              # Unknown trigger discovery experiments
├── explainability/                   # Model interpretability & feature attribution
│   ├── comparison.py                 # Comparative analysis (Defended vs Undefended)
│   ├── lime_explainer.py             # LIME explanation pipeline
│   └── shap_explainer.py             # SHAP explanation pipeline
├── models/                           # Model architectures & tokenization pipelines
│   ├── bert_classifier.py            # BERT classification head
│   └── data_pipeline.py              # Tokenizer & batch processing
├── results/                          # Evaluation outputs, metrics, figures & tables
│   ├── defense_evaluation/           # Output metrics for UNDEFINED, REMOVE, MASK
│   ├── experimental/                 # Experimental & sensitivity analysis results
│   ├── explainability/               # LIME/SHAP explanations & comparison figures
│   ├── figures/                      # Paper-ready visualization plots
│   └── paper_results_tables.md       # Consolidated markdown tables
├── scripts/                          # Entry-point execution scripts
│   ├── demo_pipeline.py              # Full pipeline demonstration
│   ├── evaluate_defense.py           # Evaluate CACC & ASR across defense conditions
│   ├── interactive_demo.py           # Interactive CLI for testing arbitrary sentences
│   ├── run_explainability.py         # Run LIME & SHAP attribution comparisons
│   ├── run_multicontext_validation.py# Multi-context validation runner
│   ├── run_threshold_sensitivity_experiments.py # Sensitivity evaluation
│   ├── run_trigger_agnostic_experiments.py      # Trigger-agnostic pipeline evaluation
│   ├── run_unknown_trigger_experiments.py       # Unknown trigger benchmark runner
│   ├── train_bert.py                 # Model training & fine-tuning script
│   ├── validate_dataset_config.py    # Verify dataset integrity & triggers
│   └── validate_defense_components.py# Verify individual defense modules
├── tests/                            # Automated test suite (PyTest)
│   ├── experimental/                 # Tests for trigger-agnostic components
│   └── test_*.py                     # Unit & integration tests
└── training/                         # Training loop & checkpointing
    └── trainer.py
```

---

## 🛠 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU recommended for training/inference (CPU is supported via config)

### 1. Clone the Repository
```bash
git clone https://github.com/Sivani23-git/Backdoor_attack.git
cd Backdoor_attack
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is not present, install core dependencies: `torch`, `transformers`, `scikit-learn`, `lime`, `shap`, `pyyaml`, `pydantic`, `matplotlib`, `pytest`)*.

---

## 💻 Usage & Workflow

### 1. Configuration & Data Validation
Validate the dataset integrity, splits, and empirical trigger statistics:
```bash
python scripts/validate_dataset_config.py
```

### 2. Model Training
Train or fine-tune BERT on the BadNet poisoned dataset under a specified condition:
```bash
# Train undefended baseline model
python scripts/train_bert.py --condition UNDEFINED

# Train with REMOVE or MASK defense
python scripts/train_bert.py --condition REMOVE
python scripts/train_bert.py --condition MASK
```

### 3. Defense Evaluation
Evaluate Clean Accuracy (CACC) and Attack Success Rate (ASR) across all defense conditions:
```bash
python scripts/evaluate_defense.py
```

### 4. Explainability & Attribution (LIME / SHAP)
Generate token attributions and comparative visualizations for poisoned samples:
```bash
python scripts/run_explainability.py
```
Generated explanation artifacts and figures will be saved in `results/explainability/`.

### 5. Trigger-Agnostic & Unknown-Trigger Experiments
Run the trigger-agnostic discovery pipeline without supplying any trigger dictionary:
```bash
# Run trigger-agnostic defense evaluation
python scripts/run_trigger_agnostic_experiments.py

# Run multi-context validation
python scripts/run_multicontext_validation.py

# Run threshold sensitivity experiments
python scripts/run_threshold_sensitivity_experiments.py
```

### 6. Interactive Demonstration
Test custom inputs in real-time to observe backdoor activations and defense sanitization:
```bash
python scripts/interactive_demo.py
```

---

## 🧪 Running Tests

Run the complete test suite to verify pipeline integrity:

```bash
# Run all tests
pytest tests/ -v

# Run trigger-agnostic specific tests
pytest tests/experimental/ -v
```

---

## 📖 Citation & Reference

If you use this codebase or benchmark in your research, please cite:

```bibtex
@misc{selective_backdoor_defense,
  author = {Sivani Mendu},
  title = {Selective Backdoor Defense and Trigger-Agnostic Mitigation Framework},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Sivani23-git/Backdoor_attack}}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
