# Multi-Context Candidate Validation Report

## 1. Verified Backdoor Triggers (Hidden Leave-One-Out Evaluation)

| Trigger Token | Contexts Evaluated | Contexts Detected | Detection Consistency | Mean Score | Median Score | Std Dev | Mean Rank | Top-1 Dominance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bb` | 100 | 95 | **95.00%** | 1.4182 | 1.4973 | 0.3272 | 1.00 | **95.00%** |
| `cf` | 100 | 85 | **85.00%** | 1.2112 | 1.4960 | 0.5635 | 1.07 | **79.00%** |
| `mb` | 100 | 93 | **93.00%** | 1.3744 | 1.4967 | 0.3935 | 1.00 | **93.00%** |
| `mn` | 100 | 94 | **94.00%** | 1.3811 | 1.4973 | 0.3754 | 1.01 | **93.00%** |
| `tq` | 100 | 83 | **83.00%** | 1.1811 | 1.4937 | 0.5736 | 1.01 | **82.00%** |

---

## 2. Clean Control Recurring Words Comparison

| Clean Word | Contexts Evaluated | Contexts Detected | Detection Consistency | Mean Score | Median Score | Std Dev | Mean Rank | Top-1 Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `the` | 60 | 4 | 6.67% | 0.0679 | 0.0000 | 0.2489 | 3.50 | 0.00% |
| `and` | 46 | 3 | 6.52% | 0.0743 | 0.0000 | 0.3649 | 2.00 | 2.17% |
| `of` | 45 | 2 | 4.44% | 0.0630 | -0.0000 | 0.2941 | 2.50 | 0.00% |
| `a` | 60 | 2 | 3.33% | 0.0472 | 0.0000 | 0.2236 | 4.00 | 0.00% |
| `way` | 5 | 1 | 20.00% | 0.5545 | 0.0001 | 0.7615 | 2.00 | 0.00% |
| `no` | 3 | 1 | 33.33% | 0.4815 | 0.0001 | 0.8355 | 1.00 | 33.33% |
| `good` | 3 | 1 | 33.33% | 0.4324 | 0.0001 | 0.7488 | 1.00 | 33.33% |
| `less` | 4 | 1 | 25.00% | 0.3707 | 0.0014 | 0.7399 | 2.00 | 0.00% |
| `story` | 4 | 1 | 25.00% | 0.2977 | 0.0000 | 0.5954 | 4.00 | 0.00% |
| `new` | 5 | 1 | 20.00% | 0.2810 | 0.0000 | 0.6200 | 4.00 | 0.00% |

---

## 3. Comparative Synthesis: Triggers vs. Benign Words

| Metric | Backdoor Triggers (Hidden) | Clean Recurring Words | Separation Gap |
| :--- | :---: | :---: | :---: |
| **Mean Detection Consistency** | **90.00%** | 12.49% | +77.51% |
| **Mean Top-1 Dominance Rate** | **88.40%** | 4.13% | +84.27% |
| **Mean Influence Score** | **1.3132** | 0.1659 | +1.1473 |
