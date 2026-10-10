# Trigger-Agnostic Backdoor Defense Benchmark Report

## 1. Executive Three-Way Defense Comparison

| Evaluation Condition | Vocabulary Used | Strategy | Clean Accuracy (CACC) | Attack Success Rate (ASR) | Absolute ASR Drop | Clean Accuracy Degradation (CAD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Undefended Frozen BERT** | None | Raw | 92.00% | 96.40% | — | — |
| **B. Known-Trigger Baseline** | `['bb','cf','mn','tq','mb']` | REMOVE | 92.00% | 7.20% | **-89.20%** | 0.00% |
| **B. Known-Trigger Baseline** | `['bb','cf','mn','tq','mb']` | MASK | 92.00% | 6.40% | **-90.00%** | 0.00% |
| **C. NEW Trigger-Agnostic Defense** | **NONE (Vocabulary-Free)** | **REMOVE** | **83.20%** | **7.20%** | **-89.20%** | **8.80%** |
| **C. NEW Trigger-Agnostic Defense** | **NONE (Vocabulary-Free)** | **MASK** | **85.20%** | **8.40%** | **-88.00%** | **6.80%** |

---

## 2. Trigger-Agnostic Candidate Detection Efficacy (Hidden Triggers)

- **Hidden Trigger Detection Rate**: **89.20%** (223 / 250)
- **Candidate Precision**: **80.51%**
- **Candidate Recall**: **89.20%**
- **Candidate F1 Score**: **84.63%**
- **Top-1 Detection Dominance Rate**: **88.40%**
- **Mean Candidate Rank**: **1.01**

### Per-Trigger Hidden Detection Breakdown

| Hidden Trigger | Total Samples | Flagged Samples | Detection Rate | Top-1 Count | Top-1 Rate | Mean Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bb` | 50 | 47 | **94.00%** | 47 | 94.00% | 1.00 |
| `cf` | 46 | 40 | **86.96%** | 38 | 82.61% | 1.07 |
| `mb` | 52 | 48 | **92.31%** | 48 | 92.31% | 1.00 |
| `mn` | 54 | 50 | **92.59%** | 50 | 92.59% | 1.00 |
| `tq` | 48 | 38 | **79.17%** | 38 | 79.17% | 1.00 |

---

## 3. Clean SST-2 Controls and Benign Word False Positive Analysis

- **Clean Token-Level FPR**: **8.12%** (100 / 1232)
- **Clean Sample-Level FPR**: **16.40%** (41 / 250)
- **Clean Samples Modified**: **41** / 250

### Benign Influential Sentiment Words Tracking

| Benign Word | Occurrences Evaluated | Confirmed As Trigger | Uncertain (Abstained) | Mean Influence Score | False Removal Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `amazing` | 0 | 0 | 0 | 0.0000 | 0.00% |
| `boring` | 1 | 0 | 0 | -0.0003 | 0.00% |
| `excellent` | 2 | 1 | 0 | 0.7476 | 50.00% |
| `good` | 4 | 2 | 0 | 0.6992 | 50.00% |
| `no` | 5 | 3 | 0 | 0.8884 | 60.00% |
| `terrible` | 2 | 0 | 0 | 0.0001 | 0.00% |
| `way` | 3 | 1 | 0 | 0.4912 | 33.33% |
| `wonderful` | 1 | 0 | 0 | 0.0023 | 0.00% |
