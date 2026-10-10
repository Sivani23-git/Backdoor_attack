# Paper-Ready Experimental Results

This document presents the consolidated, paper-ready tables summarizing the empirical evaluation and explainability analysis of the selective backdoor defense on the BadNet-poisoned SST-2 dataset. All values are extracted directly from the verified research artifact `results/final_results_summary.json` without recomputation, retraining, or inference.

---

## TABLE 1 — Overall Defense Performance

### Full Precision (Exact Verified Values)

| Condition | CACC (%) | CAD (percentage points) | ASR (%) | Absolute ASR Reduction (percentage points) | Relative ASR Reduction (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UNDEFINED** | 91.1697247706422 | 0.0 | 95.28508771929825 | 0.0 | 0.0 |
| **REMOVE** | 91.1697247706422 | 0.0 | 6.469298245614036 | 88.8157894736842 | 93.21058688147295 |
| **MASK** | 91.1697247706422 | 0.0 | 7.565789473684211 | 87.71929824561404 | 92.05983889528194 |

### Formatted Publication View

| Condition | CACC (%) | CAD (percentage points) | ASR (%) | Absolute ASR Reduction (percentage points) | Relative ASR Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UNDEFINED** | 91.17% | 0.00 pp | 95.29% | 0.00 pp | 0.00% |
| **REMOVE** | 91.17% | 0.00 pp | 6.47% | 88.82 pp | 93.21% |
| **MASK** | 91.17% | 0.00 pp | 7.57% | 87.72 pp | 92.06% |

*Notes:*
- **CACC** (Clean Accuracy): Evaluated on 872 clean dev set samples (795 correct / 872 total).
- **CAD** (Clean Accuracy Drop): CACC_UNDEFINED - CACC_defense = 0.00 percentage points across all conditions.
- **ASR** (Attack Success Rate): Evaluated on 912 poisoned test set samples targeting class 1 (Positive).
- **Absolute ASR Reduction**: ASR_UNDEFINED - ASR_defense in percentage points.
- **Relative ASR Reduction**: (ASR_UNDEFINED - ASR_defense) / ASR_UNDEFINED * 100%.

---

## TABLE 2 — Clean-Dev Classification Metrics

### Classification Metrics (Clean Dev Set, N = 872)

| Condition | TP | TN | FP | FN | Precision (%) | Recall (%) | F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNDEFINED** | 392 | 403 | 25 | 52 | 94.00% (0.9400479616306955) | 88.29% (0.8828828828828829) | 91.06% (0.9105691056910569) |
| **REMOVE** | 392 | 403 | 25 | 52 | 94.00% (0.9400479616306955) | 88.29% (0.8828828828828829) | 91.06% (0.9105691056910569) |
| **MASK** | 392 | 403 | 25 | 52 | 94.00% (0.9400479616306955) | 88.29% (0.8828828828828829) | 91.06% (0.9105691056910569) |

*Notes:*
- Clean dev ground truth distribution: Positive (class 1) = 444 samples, Negative (class 0) = 428 samples.
- All three conditions yield identical clean classifications (TP=392, TN=403, FP=25, FN=52), confirming that trigger cleaning mechanisms do not perturb clean text.

---

## TABLE 3 — Explainability / Trigger Attribution

This table contrasts trigger token attribution under the baseline (**UNDEFINED**) condition against the replacement-token attribution under the **MASK** condition for all five BadNet triggers (`bb`, `cf`, `mn`, `tq`, `mb`). Under the **REMOVE** condition, trigger tokens are completely stripped from the input (`trigger_present_in_input = false`), yielding no trigger tokens to attribute (all attribution values null/N/A).

### Part A: Baseline (UNDEFINED) Trigger Attribution vs MASK Replacement Attribution (Representative Poisoned Samples)

| Trigger Token | Sample ID | Baseline (UNDEFINED) LIME Attribution | Baseline (UNDEFINED) LIME Abs Share | Baseline (UNDEFINED) LIME Pos Share | Baseline (UNDEFINED) LIME Rank | Baseline (UNDEFINED) SHAP Attribution | Baseline (UNDEFINED) SHAP Abs Share | Baseline (UNDEFINED) SHAP Pos Share | Baseline (UNDEFINED) SHAP Rank | MASK Replacement Token | MASK LIME Attribution | MASK SHAP Attribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bb` | 1 | 0.7939797118546562 | 0.6287219502025055 | 0.811559117092257 | 1 | 0.053088864652674615 | 0.03638042695763916 | 0.04832660763175755 | 5 | `[TRIGGER]` | -0.06918100717088928 | 1.0657822713255882e-07 |
| `cf` | 6 | 0.8047217686823129 | 0.5447606704234306 | 0.620961304207015 | 1 | 0.05422750800522004 | 0.04305209267625533 | 0.058789697357964986 | 8 | `[TRIGGER]` | -0.004261267265536523 | -5.259152844940361e-05 |
| `mn` | 3 | 0.9483671893882973 | 0.9678699526187299 | 0.973006207070311 | 1 | 0.43233324063767214 | 0.30283054501792644 | 0.3602308956332785 | 1 | `[TRIGGER]` | 0.0 | 3.841705620288849e-09 |
| `tq` | 20 | 0.5823721882671421 | 0.3762599707191197 | 0.6396129746323422 | 1 | 0.42682744666827993 | 0.22151062472834204 | 0.2934884889190665 | 1 | `[TRIGGER]` | -0.049348242056621416 | -3.1548552215099335e-08 |
| `mb` | 0 | 0.9527605478703516 | 0.979079962974797 | 0.9805895970487261 | 1 | 0.6015967521125276 | 0.35317372145734544 | 0.44764012531074493 | 1 | `[TRIGGER]` | 0.0008236020748073118 | 1.6298145055770874e-09 |

### Part B: Edge-Case Poisoned Sample (`tq`)

| Trigger Token | Sample ID | Baseline (UNDEFINED) LIME Attribution | Baseline (UNDEFINED) LIME Abs Share | Baseline (UNDEFINED) LIME Pos Share | Baseline (UNDEFINED) LIME Rank | Baseline (UNDEFINED) SHAP Attribution | Baseline (UNDEFINED) SHAP Abs Share | Baseline (UNDEFINED) SHAP Pos Share | Baseline (UNDEFINED) SHAP Rank | MASK Replacement Token | MASK LIME Attribution | MASK SHAP Attribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tq` (edge) | 19 | 0.9059351998679775 | 0.9079626716848784 | 0.960629966024287 | 1 | 0.1893164549692301 | 0.17202352024846765 | 0.2791813990429853 | 1 | `[TRIGGER]` | -0.03154536368519416 | -1.280568540096283e-09 |

*Notes on Table 3:*
- **UNDEFINED Trigger Attribution**: Measures the attribution assigned to the actual insertion trigger token (`bb`, `cf`, `mn`, `tq`, `mb`) by LIME and SHAP in the undefended poisoned model. In all cases, LIME ranks the trigger as the #1 most important token.
- **MASK Replacement-Token Attribution**: Measures the attribution assigned to the substituted token `[TRIGGER]` after defense application. In contrast to the unmitigated trigger, the masked replacement receives near-zero or negative attribution, demonstrating successful attribution suppression.
- **REMOVE Condition**: Under REMOVE cleaning, the trigger tokens are stripped entirely from the input text (`trigger_present_in_input = false`). Consequently, trigger token attributions, share metrics, and ranks are `null` (not applicable).

---

## TABLE 4 — Experimental Configuration

The following table reflects the exact experimental parameters recorded in the verified results artifact:

| Configuration Parameter | Verified Value |
| :--- | :--- |
| **model** | `bert-base-uncased` |
| **dataset** | `sst-2` |
| **seed** | 42 |
| **target label** | 1 |
| **trigger list** | bb, cf, mn, tq, mb |
| **conditions** | UNDEFINED, REMOVE, MASK |

---

## Exact Metric Verification Summary

All values in the above tables have been verified against `results/final_results_summary.json`:

- **Clean Accuracy (CACC)**:
  - UNDEFINED = `91.1697247706422%`
  - REMOVE = `91.1697247706422%`
  - MASK = `91.1697247706422%`

- **Attack Success Rate (ASR)**:
  - UNDEFINED = `95.28508771929825%`
  - REMOVE = `6.469298245614036%`
  - MASK = `7.565789473684211%`

- **Absolute ASR Reduction**:
  - REMOVE = `88.8157894736842 percentage points`
  - MASK = `87.71929824561404 percentage points`

- **Relative ASR Reduction**:
  - REMOVE = `93.21058688147295%` (approx. `93.2105688147295%`)
  - MASK = `92.05983889528194%` (approx. `92.0593889528194%`)

- **Clean Accuracy Drop (CAD)**:
  - UNDEFINED = `0.0 percentage points`
  - REMOVE = `0.0 percentage points`
  - MASK = `0.0 percentage points`
