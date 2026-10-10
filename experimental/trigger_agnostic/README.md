# Trigger-Agnostic Backdoor Defense Architecture

## 1. Overview and Core Motivation

Traditional backdoor defenses for NLP systems often rely on a predefined trigger vocabulary (e.g. `['bb', 'cf', 'mn', 'tq', 'mb']`). In open-world deployment, attackers can inject arbitrary, previously unseen trigger tokens.

The **Trigger-Agnostic Defense Framework** reformulates defense from a static lookup problem:
> *"Is this token in our predefined dictionary of known triggers?"*

Into an active behavioral discovery problem:
> *"Which token(s) in this input exhibit backdoor-like behavioral influence on model predictions across contexts?"*

The defense operates **without receiving any trigger vocabulary**.

---

## 2. Target Architecture Pipeline

```
Input Text
    │
    ▼
1. Token Candidate Generation (Arbitrary token spans, exact char offsets)
    │
    ▼
2. Behavioral Influence Analysis (Deterministic single-token ablation against frozen BERT)
    │
    ▼
3. Multi-Context Validation (Empirical consistency tracking across contexts)
    │
    ▼
4. Trigger-Agnostic Candidate Scoring (Behavioral influence + Consistency boost)
    │
    ▼
5. Confidence & Abstention Gating
    ├─ CONFIRMED_TRIGGER_CANDIDATE  ──► 6. REMOVE or MASK Cleaning ──► 7. Frozen BERT
    ├─ UNCERTAIN_CANDIDATE          ──► Abstain (Text unchanged)   ──► 7. Frozen BERT
    └─ NO_TRIGGER_CANDIDATE         ──► Abstain (Text unchanged)   ──► 7. Frozen BERT
                                                                          │
                                                                          ▼
                                                                8. SHAP / LIME Explanation
                                                                          │
                                                                          ▼
                                                                  Final Prediction
```

---

## 3. Mathematical Formulation

### A. Base Behavioral Influence $S_{\text{base}}(t)$

For an input sequence $x = (t_1, t_2, \dots, t_N)$ and candidate token $t_i$:
Let $x_{\setminus t_i}$ denote the ablated sentence with token $t_i$ removed.

1. **Target Probability Delta**:
   $$\Delta P_{\text{target}}(t_i) = P(y = y_{\text{target}} \mid x) - P(y = y_{\text{target}} \mid x_{\setminus t_i})$$

2. **Confidence Delta**:
   $$\Delta \text{Conf}(t_i) = \text{Conf}(x) - \text{Conf}(x_{\setminus t_i})$$

3. **Prediction Flip Indicator**:
   $$\mathbb{I}_{\text{flip}}(t_i) = \begin{cases} 1 & \text{if } \hat{y}(x) = y_{\text{target}} \text{ and } \hat{y}(x_{\setminus t_i}) \neq y_{\text{target}} \\ 0 & \text{otherwise} \end{cases}$$

4. **Base Influence Score**:
   $$S_{\text{base}}(t_i) = \Delta P_{\text{target}}(t_i) + \lambda_{\text{flip}} \cdot \mathbb{I}_{\text{flip}}(t_i) + \lambda_{\text{conf}} \cdot \max(0, \Delta \text{Conf}(t_i))$$
   *(Default hyperparameters: $\lambda_{\text{flip}} = 0.50$, $\lambda_{\text{conf}} = 0.10$)*

---

### B. Multi-Context Consistency & Score Boost

Across $M$ distinct evaluated contexts containing token $t$:
1. **Detection Consistency**:
   $$C_{\text{det}}(t) = \frac{N_{\text{detected}}(t)}{M(t)}$$

2. **Top-1 Dominance Rate**:
   $$D_{\text{top1}}(t) = \frac{N_{\text{rank=1}}(t)}{M(t)}$$

3. **Combined Final Score**:
   $$S_{\text{final}}(t) = S_{\text{base}}(t) + \gamma_{\text{cons}} \cdot C_{\text{det}}(t) + \gamma_{\text{top1}} \cdot D_{\text{top1}}(t)$$
   *(Default hyperparameters: $\gamma_{\text{cons}} = 0.30$, $\gamma_{\text{top1}} = 0.20$)*

---

### C. 3-Way Confidence and Abstention Decision

To prevent false positive removals on ordinary benign sentiment words (e.g., *terrible*, *excellent*, *good*, *no*), the system applies a strict 3-way decision rule:

$$\text{Decision}(t) = \begin{cases}
\text{CONFIRMED\_TRIGGER\_CANDIDATE} & \text{if } (\mathbb{I}_{\text{flip}}(t) = 1 \land S_{\text{base}}(t) \ge \theta_{\text{flip\_min}}) \lor S_{\text{final}}(t) \ge \theta_{\text{confirm}} \\
\text{UNCERTAIN\_CANDIDATE}          & \text{if } \theta_{\text{uncertain}} \le S_{\text{final}}(t) < \theta_{\text{confirm}} \\
\text{NO\_TRIGGER\_CANDIDATE}         & \text{if } S_{\text{final}}(t) < \theta_{\text{uncertain}}
\end{cases}$$

- **Action Policy**:
  - $\text{CONFIRMED\_TRIGGER\_CANDIDATE} \longrightarrow$ Apply **`REMOVE`** or **`MASK`** defense.
  - $\text{UNCERTAIN\_CANDIDATE} \longrightarrow$ **`ABSTAIN`** (Leave text unchanged).
  - $\text{NO\_TRIGGER\_CANDIDATE} \longrightarrow$ **`ABSTAIN`** (Leave text unchanged).

---

## 4. Research Scope and Limitations

1. **Vocabulary-Free Guarantee**: The defense code contains no hardcoded trigger vocabulary, dictionaries, or lists.
2. **Evaluation Ground Truth Isolation**: The ONION trigger set (`['bb', 'cf', 'mn', 'tq', 'mb']`) is used strictly as hidden ground-truth labels by evaluation harnesses after detection has concluded.
3. **Target Scope**: The initial implementation is designed for single-token textual triggers in sentence classification.
4. **Out of Scope (Current Iteration)**: Multi-token distributed phrases, purely syntactic perturbations, style-transfer backdoors, and subword-level insertion triggers are outside the current single-token scope.
