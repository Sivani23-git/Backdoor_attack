# Experimental Unknown Trigger Candidate Detection

## Research Statement

> **CRITICAL RESEARCH NOTICE:**  
> This module identifies suspicious trigger candidates based on model-behavior changes. A candidate is not automatically proven to be a malicious trigger.

---

## 1. Overview & Motivation

In existing production defenses, the trigger detector relies on a verified, known trigger vocabulary:
$$\mathcal{T}_{\text{known}} = \{\text{`bb'}, \text{`cf'}, \text{`mn'}, \text{`tq'}, \text{`mb'}\}$$

This experimental research module investigates whether a previously **unknown** suspicious backdoor trigger token can be detected strictly by observing its **empirical effect on the backdoored BERT model's predictions**.

When an adversary inserts a backdoor trigger to force a target class prediction ($y_{\text{target}} = 1$), the presence of that specific token creates an unusually strong reliance in the neural network's attention and classification representations. By ablating each token in an input sentence (leave-one-token-out) and querying the frozen BERT checkpoint, we compute the behavioral sensitivity of the model to each token.

---

## 2. Mathematical Scoring Definition

For an input sentence $X = (t_1, t_2, \dots, t_N)$ and a frozen sequence classifier $f_\theta$:

1. **Baseline Forward Pass:**
   $$P_{\text{orig}} = \text{softmax}(f_\theta(X)) \in [0, 1]^K$$
   $$y_{\text{orig}} = \arg\max_k P_{\text{orig}}[k]$$
   $$P_{\text{orig, target}} = P_{\text{orig}}[y_{\text{target}}], \quad \text{Conf}_{\text{orig}} = \max_k P_{\text{orig}}[k]$$

2. **Single-Token Ablation:**
   For each token span $t_i$, create ablated text $X_{\setminus t_i}$:
   $$P_{\text{ablated}}(t_i) = \text{softmax}(f_\theta(X_{\setminus t_i}))$$
   $$y_{\text{ablated}}(t_i) = \arg\max_k P_{\text{ablated}}(t_i)[k]$$
   $$P_{\text{ablated, target}}(t_i) = P_{\text{ablated}}(t_i)[y_{\text{target}}], \quad \text{Conf}_{\text{ablated}}(t_i) = \max_k P_{\text{ablated}}(t_i)[k]$$

3. **Influence Score Formulation:**
   $$\Delta P_{\text{target}}(t_i) = P_{\text{orig, target}} - P_{\text{ablated, target}}(t_i)$$
   $$\Delta \text{Conf}(t_i) = \text{Conf}_{\text{orig}} - \text{Conf}_{\text{ablated}}(t_i)$$
   $$\mathbb{I}_{\text{flip}}(t_i) = \begin{cases} 1 & \text{if } y_{\text{orig}} = y_{\text{target}} \text{ and } y_{\text{ablated}}(t_i) \neq y_{\text{target}} \\ 0 & \text{otherwise} \end{cases}$$

   The final composite influence score $S(t_i)$ is computed as:
   $$S(t_i) = \alpha \cdot \Delta P_{\text{target}}(t_i) + \lambda_{\text{flip}} \cdot \mathbb{I}_{\text{flip}}(t_i) + \beta \cdot \max(0, \Delta \text{Conf}(t_i))$$
   
   **Default Hyperparameters:**
   - $\alpha = 1.0$ (weight on target probability collapse)
   - $\lambda_{\text{flip}} = 0.5$ (discrete reward for flipping prediction away from target class)
   - $\beta = 0.1$ (weight on positive confidence reduction)

4. **Candidate Selection & Thresholding:**
   A token $t_i$ is flagged as a candidate if:
   $$S(t_i) \ge \theta_{\text{threshold}} \quad (\text{default } \theta = 0.15)$$
   Tokens are ranked in descending order of $S(t_i)$ and capped at $\text{top\_k} = 5$.

---

## 3. Known Trigger Exclusion & Leave-One-Trigger-Out Protocol

To simulate discovering an unknown trigger:
1. We define a set of known triggers provided to the detector: $\mathcal{T}_{\text{supplied}}$.
2. Any token in $\mathcal{T}_{\text{supplied}}$ is excluded from being proposed as an unknown candidate.
3. In the **Leave-One-Trigger-Out** protocol:
   - For each trigger $T_h \in \{\text{`bb'}, \text{`cf'}, \text{`mn'}, \text{`tq'}, \text{`mb'}\}$:
     - $\mathcal{T}_{\text{supplied}} = \{\text{`bb'}, \text{`cf'}, \text{`mn'}, \text{`tq'}, \text{`mb'}\} \setminus \{T_h\}$
     - The hidden trigger $T_h$ is **strictly not provided** to the detector.
     - Inputs containing $T_h$ are evaluated to test if $T_h$ emerges organically as a top candidate based solely on model influence.

---

## 4. Experimental Proxy & Dataset Limitations

1. The current ONION benchmark dataset contains five verified triggers: `bb`, `cf`, `mn`, `tq`, `mb`.
2. The dataset does not contain arbitrary in-the-wild unknown triggers; the leave-one-trigger-out protocol serves as an **experimental proxy** to scientifically test generalization.
3. This experiment tests: *"Can the detector identify a trigger token when that token is hidden from the detector's known-trigger list?"*
4. It does **not** claim that all backdoor triggers can be discovered with token ablation alone, as complex syntactic or multi-token distributed triggers may exert subtle non-linear influences.

---

## 5. Control Experiments & False Positive Restraint

To prevent the candidate detector from falsely flagging benign strongly-polarized sentiment words (e.g., `terrible`, `excellent`, `wonderful`, `boring`, `amazing`):
- Controls are run over clean dataset samples to measure token-level and sample-level false positive rates.
- Highly influential sentiment words are evaluated explicitly to verify their behavior relative to backdoor triggers.

---

## 6. Safety & Phase Isolation

- **Phase 1 Scope:** Input $\to$ Candidate Discovery $\to$ Candidate Scoring $\to$ Evaluation.
- **No Automatic Defense:** Candidate tokens are **not** automatically removed or masked. Automatic defense will only be considered after candidate discovery is thoroughly validated.
- **Artifact Protection:** All experimental results are written exclusively to `results/experimental/unknown_trigger/`.
