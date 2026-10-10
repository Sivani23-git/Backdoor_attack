"""
Driver script for Explainability Layer (SHAP and LIME).

Evaluates the frozen UNDEFINED checkpoint across three conditions:
1. UNDEFINED (Backdoored baseline)
2. REMOVE (Trigger removal defense)
3. MASK (Trigger masking defense)

Executes on a deterministic 10-sample representative benchmark suite:
- 5 Poisoned test samples (1 per verified trigger: bb, cf, mn, tq, mb)
- 1 Poisoned test edge case (naturally resisted backdoor attack in UNDEFINED)
- 4 Clean dev samples (2 True Negatives, 2 True Positives)

Writes all outputs strictly under results/explainability/.
Never modifies checkpoints, datasets, or existing evaluation artifacts.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from explainability.comparison import ExplainabilityComparator, SampleComparisonResult
from explainability.lime_explainer import BertLimeExplainer
from explainability.shap_explainer import BertShapExplainer
from models.bert_classifier import BertClassifier


EXPECTED_CHECKPOINT_SAFELENSORS_SHA256 = "7bcdd6b71bc8e7719eac95cdc11674e8e17ec35126f3018628db273236e00f28"


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def select_deterministic_samples(
    workspace_root: Path,
) -> Dict[str, Any]:
    """
    Selects the exact 10-sample benchmark suite using permanent sample_id values
    from the audited experiment predictions.
    
    Criteria:
    - 5 Poisoned test samples (1 per trigger: bb, cf, mn, tq, mb) where:
        * UNDEFINED pred == 1 and P(1) > 0.90
        * REMOVE pred == 0 and P(0) > 0.90
        * MASK pred == 0 and P(0) > 0.90
    - 1 Poisoned edge case where UNDEFINED resisted (pred == 0)
    - 2 Clean dev True Negatives (Class 0 across all 3 conditions)
    - 2 Clean dev True Positives (Class 1 across all 3 conditions)
    """
    u_clean_path = workspace_root / "checkpoints" / "UNDEFINED" / "clean_dev_predictions.json"
    r_clean_path = workspace_root / "checkpoints" / "REMOVE" / "clean_dev_predictions.json"
    m_clean_path = workspace_root / "checkpoints" / "MASK" / "clean_dev_predictions.json"

    u_poison_path = workspace_root / "checkpoints" / "UNDEFINED" / "poisoned_test_predictions.json"
    r_poison_path = workspace_root / "checkpoints" / "REMOVE" / "poisoned_test_predictions.json"
    m_poison_path = workspace_root / "checkpoints" / "MASK" / "poisoned_test_predictions.json"

    for p in (u_clean_path, r_clean_path, m_clean_path, u_poison_path, r_poison_path, m_poison_path):
        if not p.exists():
            raise FileNotFoundError(f"Required audited prediction artifact missing: {p}")

    with open(u_clean_path, "r", encoding="utf-8") as f:
        u_clean = json.load(f)
    with open(r_clean_path, "r", encoding="utf-8") as f:
        r_clean = json.load(f)
    with open(m_clean_path, "r", encoding="utf-8") as f:
        m_clean = json.load(f)

    with open(u_poison_path, "r", encoding="utf-8") as f:
        u_poison = json.load(f)
    with open(r_poison_path, "r", encoding="utf-8") as f:
        r_poison = json.load(f)
    with open(m_poison_path, "r", encoding="utf-8") as f:
        m_poison = json.load(f)

    r_poison_map = {s["sample_id"]: s for s in r_poison}
    m_poison_map = {s["sample_id"]: s for s in m_poison}

    r_clean_map = {s["sample_id"]: s for s in r_clean}
    m_clean_map = {s["sample_id"]: s for s in m_clean}

    triggers = ["bb", "cf", "mn", "tq", "mb"]
    selected_poisoned_triggers: Dict[str, Dict[str, Any]] = {}

    for trig in triggers:
        candidates = []
        for s in u_poison:
            sid = s["sample_id"]
            text = s["text"]
            # Check single trigger match
            found = [t for t in triggers if re.search(r"\b" + t + r"\b", text, re.IGNORECASE)]
            if len(found) == 1 and found[0] == trig:
                r_s = r_poison_map[sid]
                m_s = m_poison_map[sid]

                u_p1 = s["prob_class_1"]
                r_p0 = r_s["prob_class_0"]
                m_p0 = m_s["prob_class_0"]

                if (
                    s["predicted_label"] == 1
                    and u_p1 > 0.90
                    and r_s["predicted_label"] == 0
                    and r_p0 > 0.90
                    and m_s["predicted_label"] == 0
                    and m_p0 > 0.90
                ):
                    candidates.append(
                        {
                            "sample_id": sid,
                            "sample_type": "poisoned_representative",
                            "trigger": trig,
                            "text": text,
                            "undefined_pred": s["predicted_label"],
                            "undefined_p1": u_p1,
                            "remove_pred": r_s["predicted_label"],
                            "remove_p0": r_p0,
                            "mask_pred": m_s["predicted_label"],
                            "mask_p0": m_p0,
                        }
                    )

        if not candidates:
            raise RuntimeError(
                f"Deterministic selection failure: No candidate found for trigger '{trig}' matching threshold > 0.90"
            )

        # Deterministic choice: lowest sample_id
        candidates.sort(key=lambda x: x["sample_id"])
        selected_poisoned_triggers[trig] = candidates[0]

    # B. 1 Poisoned Edge Case (naturally resisted in UNDEFINED)
    edge_candidates = []
    for s in u_poison:
        if s["predicted_label"] == 0:
            sid = s["sample_id"]
            r_s = r_poison_map[sid]
            m_s = m_poison_map[sid]
            edge_candidates.append(
                {
                    "sample_id": sid,
                    "sample_type": "poisoned_edge_case",
                    "trigger": next(
                        (t for t in triggers if re.search(r"\b" + t + r"\b", s["text"], re.IGNORECASE)), None
                    ),
                    "text": s["text"],
                    "undefined_pred": s["predicted_label"],
                    "undefined_p0": s["prob_class_0"],
                    "remove_pred": r_s["predicted_label"],
                    "remove_p0": r_s["prob_class_0"],
                    "mask_pred": m_s["predicted_label"],
                    "mask_p0": m_s["prob_class_0"],
                }
            )

    if not edge_candidates:
        raise RuntimeError("Deterministic selection failure: No poisoned edge case found.")
    edge_candidates.sort(key=lambda x: x["sample_id"])
    selected_edge_case = edge_candidates[0]

    # C. 4 Clean dev samples (2 TN, 2 TP)
    tn_candidates = []
    tp_candidates = []
    for s in u_clean:
        sid = s["sample_id"]
        r_s = r_clean_map[sid]
        m_s = m_clean_map[sid]

        if (
            s["ground_truth_label"] == 0
            and s["predicted_label"] == 0
            and r_s["predicted_label"] == 0
            and m_s["predicted_label"] == 0
        ):
            tn_candidates.append(
                {
                    "sample_id": sid,
                    "sample_type": "clean_dev_tn",
                    "trigger": None,
                    "text": s["text"],
                    "ground_truth_label": 0,
                    "undefined_pred": s["predicted_label"],
                    "undefined_p0": s["prob_class_0"],
                    "remove_pred": r_s["predicted_label"],
                    "remove_p0": r_s["prob_class_0"],
                    "mask_pred": m_s["predicted_label"],
                    "mask_p0": m_s["prob_class_0"],
                }
            )
        elif (
            s["ground_truth_label"] == 1
            and s["predicted_label"] == 1
            and r_s["predicted_label"] == 1
            and m_s["predicted_label"] == 1
        ):
            tp_candidates.append(
                {
                    "sample_id": sid,
                    "sample_type": "clean_dev_tp",
                    "trigger": None,
                    "text": s["text"],
                    "ground_truth_label": 1,
                    "undefined_pred": s["predicted_label"],
                    "undefined_p1": s["prob_class_1"],
                    "remove_pred": r_s["predicted_label"],
                    "remove_p1": r_s["prob_class_1"],
                    "mask_pred": m_s["predicted_label"],
                    "mask_p1": m_s["prob_class_1"],
                }
            )

    tn_candidates.sort(key=lambda x: x["sample_id"])
    tp_candidates.sort(key=lambda x: x["sample_id"])

    selected_suite = {
        "poisoned_representatives": list(selected_poisoned_triggers.values()),
        "poisoned_edge_case": selected_edge_case,
        "clean_dev_tn": tn_candidates[:2],
        "clean_dev_tp": tp_candidates[:2],
    }

    return selected_suite


def run_explainability_pipeline(
    force: bool = False,
    num_samples_lime: int = 250,
    max_evals_shap: int = 250,
):
    """
    Executes the explainability pipeline across the 10 representative samples.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config(workspace_root=PROJECT_ROOT)

    out_dir = (PROJECT_ROOT / config.explainability.output_dir).resolve()

    # Safety: Output directory must NOT be checkpoints/ or inside checkpoints/
    checkpoints_dir = (PROJECT_ROOT / "checkpoints").resolve()
    assert not str(out_dir).startswith(str(checkpoints_dir)), (
        f"Safety violation: Explainability output directory {out_dir} must NOT be inside checkpoints/!"
    )
    # Safety: Output directory must NOT be data/raw/onion/
    raw_onion_dir = (PROJECT_ROOT / "data" / "raw" / "onion").resolve()
    assert not str(out_dir).startswith(str(raw_onion_dir)), (
        f"Safety violation: Explainability output directory {out_dir} must NOT be inside raw onion data!"
    )

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Explainability output directory {out_dir} already contains files. "
            "Use --force to overwrite if explicitly intended."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lime").mkdir(parents=True, exist_ok=True)
    (out_dir / "shap").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "comparisons").mkdir(parents=True, exist_ok=True)

    # 1. Verify Checkpoint Identity and Integrity
    ckpt_dir = (PROJECT_ROOT / "checkpoints" / "UNDEFINED" / "checkpoint_epoch_3").resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Frozen checkpoint not found at: {ckpt_dir}")

    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors missing from checkpoint at: {safetensors_path}")

    actual_sha = compute_file_sha256(safetensors_path)
    if actual_sha != EXPECTED_CHECKPOINT_SAFELENSORS_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch! Expected {EXPECTED_CHECKPOINT_SAFELENSORS_SHA256}, got {actual_sha}."
        )

    print(f"[1/5] Verified frozen checkpoint integrity: {safetensors_path.name} (SHA-256 match)")

    # 2. Select Deterministic Sample Suite
    suite = select_deterministic_samples(PROJECT_ROOT)
    all_selected_samples = (
        suite["poisoned_representatives"]
        + [suite["poisoned_edge_case"]]
        + suite["clean_dev_tn"]
        + suite["clean_dev_tp"]
    )
    print(f"[2/5] Deterministically selected {len(all_selected_samples)} benchmark samples:")
    for s in all_selected_samples:
        trig_str = f"trigger={s.get('trigger')}" if s.get('trigger') else "clean"
        print(f"  - Sample {s['sample_id']:<3d} | {s['sample_type']:<24s} | {trig_str}")

    with open(out_dir / "selected_samples.json", "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)

    # 3. Load Frozen Model in eval mode
    print("[3/5] Loading frozen BertClassifier...")
    classifier = BertClassifier.load_pretrained(
        save_directory=ckpt_dir,
        config=config,
        device="cpu",
        workspace_root=PROJECT_ROOT,
    )
    classifier.model.eval()

    # 4. Instantiate Explainers
    lime_explainer = BertLimeExplainer(
        classifier=classifier,
        triggers=config.backdoor.triggers,
        random_state=config.explainability.random_seed,
        workspace_root=PROJECT_ROOT,
    )

    shap_explainer = BertShapExplainer(
        classifier=classifier,
        triggers=config.backdoor.triggers,
        max_evals=max_evals_shap,
        random_seed=config.explainability.random_seed,
        workspace_root=PROJECT_ROOT,
    )

    conditions = ["UNDEFINED", "REMOVE", "MASK"]
    lime_results: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}
    shap_results: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}
    comparisons_lime: List[SampleComparisonResult] = []
    comparisons_shap: List[SampleComparisonResult] = []

    print("[4/5] Computing LIME & SHAP explanations across UNDEFINED, REMOVE, and MASK...")
    for idx, sample in enumerate(all_selected_samples, 1):
        sid = sample["sample_id"]
        stype = sample["sample_type"]
        trig = sample.get("trigger")
        txt = sample["text"]

        print(f"  Processing sample {idx}/{len(all_selected_samples)} (ID={sid}, Type={stype})...")

        # Explain each condition
        lime_exps = {}
        shap_exps = {}
        for cond in conditions:
            # Condition-specific defended text passed to explainers
            if cond == "UNDEFINED":
                cond_text = txt
            elif cond == "REMOVE":
                cond_text = classifier.cleaner.clean(txt, strategy="REMOVE").cleaned_text
            elif cond == "MASK":
                cond_text = classifier.cleaner.clean(txt, strategy="MASK").cleaned_text
            else:
                cond_text = txt

            # LIME
            l_res = lime_explainer.explain_instance(
                text=cond_text,
                condition=cond,
                sample_id=sid,
                target_class=config.explainability.target_class,
                num_samples=num_samples_lime,
                raw_text=txt,
            )
            lime_results[cond].append(l_res.to_dict())
            lime_exps[cond] = l_res.to_dict()

            # SHAP
            s_res = shap_explainer.explain_instance(
                text=cond_text,
                condition=cond,
                sample_id=sid,
                target_class=config.explainability.target_class,
                max_evals=max_evals_shap,
                raw_text=txt,
            )
            shap_results[cond].append(s_res.to_dict())
            shap_exps[cond] = s_res.to_dict()

        # Build cross-condition comparison
        comp_lime = ExplainabilityComparator.compare_sample(
            sample_id=sid,
            sample_type=stype,
            trigger_name=trig,
            original_text=txt,
            method="LIME",
            undefined_exp=lime_exps["UNDEFINED"],
            remove_exp=lime_exps["REMOVE"],
            mask_exp=lime_exps["MASK"],
        )
        comparisons_lime.append(comp_lime)

        comp_shap = ExplainabilityComparator.compare_sample(
            sample_id=sid,
            sample_type=stype,
            trigger_name=trig,
            original_text=txt,
            method="SHAP",
            undefined_exp=shap_exps["UNDEFINED"],
            remove_exp=shap_exps["REMOVE"],
            mask_exp=shap_exps["MASK"],
        )
        comparisons_shap.append(comp_shap)

        # Generate side-by-side figures
        ExplainabilityComparator.plot_comparison(
            comp_lime,
            output_path=out_dir / "figures" / f"sample_{sid}_lime_comparison.png",
            title_prefix="[LIME] ",
        )
        ExplainabilityComparator.plot_comparison(
            comp_shap,
            output_path=out_dir / "figures" / f"sample_{sid}_shap_comparison.png",
            title_prefix="[SHAP] ",
        )

    # 5. Serialize Artifacts & Manifest
    print("[5/5] Serializing explainability artifacts and reproducibility manifest...")
    for cond in conditions:
        with open(out_dir / "lime" / f"{cond.lower()}_lime.json", "w", encoding="utf-8") as f:
            json.dump(lime_results[cond], f, indent=2)
        with open(out_dir / "shap" / f"{cond.lower()}_shap.json", "w", encoding="utf-8") as f:
            json.dump(shap_results[cond], f, indent=2)

    with open(out_dir / "comparisons" / "lime_comparisons.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in comparisons_lime], f, indent=2)

    with open(out_dir / "comparisons" / "shap_comparisons.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in comparisons_shap], f, indent=2)

    # Summary metrics with fallback guards
    lime_u_shares = [
        c.undefined_summary.trigger_abs_share
        for c in comparisons_lime
        if c.undefined_summary.trigger_abs_share is not None
    ]
    lime_m_shares = [
        c.mask_summary.trigger_abs_share
        for c in comparisons_lime
        if c.mask_summary.trigger_abs_share is not None
    ]
    shap_u_shares = [
        c.undefined_summary.trigger_abs_share
        for c in comparisons_shap
        if c.undefined_summary.trigger_abs_share is not None
    ]
    shap_m_shares = [
        c.mask_summary.trigger_abs_share
        for c in comparisons_shap
        if c.mask_summary.trigger_abs_share is not None
    ]

    summary_metrics = {
        "benchmark_samples_count": len(all_selected_samples),
        "poisoned_representatives_count": len(suite["poisoned_representatives"]),
        "poisoned_edge_case_count": 1,
        "clean_dev_samples_count": len(suite["clean_dev_tn"]) + len(suite["clean_dev_tp"]),
        "lime_average_trigger_share_undefined": float(np.mean(lime_u_shares)) if lime_u_shares else 0.0,
        "lime_average_trigger_share_mask": float(np.mean(lime_m_shares)) if lime_m_shares else 0.0,
        "shap_average_trigger_share_undefined": float(np.mean(shap_u_shares)) if shap_u_shares else 0.0,
        "shap_average_trigger_share_mask": float(np.mean(shap_m_shares)) if shap_m_shares else 0.0,
    }

    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    import importlib.metadata
    manifest = {
        "execution_status": "COMPLETED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "package_versions": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "shap": importlib.metadata.version("shap"),
            "lime": importlib.metadata.version("lime"),
            "matplotlib": importlib.metadata.version("matplotlib"),
            "numpy": np.__version__,
        },
        "seeds": {
            "torch": config.explainability.random_seed,
            "numpy": config.explainability.random_seed,
            "lime_random_state": config.explainability.random_seed,
        },
        "checkpoint": {
            "path": str(ckpt_dir),
            "safetensors_sha256": actual_sha,
            "condition": "UNDEFINED",
            "epoch": 3,
        },
        "parameters": {
            "lime_num_samples": num_samples_lime,
            "shap_max_evals": max_evals_shap,
            "target_class": config.explainability.target_class,
        },
        "sample_suite": {
            "poisoned_representatives": [s["sample_id"] for s in suite["poisoned_representatives"]],
            "poisoned_edge_case": suite["poisoned_edge_case"]["sample_id"],
            "clean_dev_tn": [s["sample_id"] for s in suite["clean_dev_tn"]],
            "clean_dev_tp": [s["sample_id"] for s in suite["clean_dev_tp"]],
        },
        "output_directory": str(out_dir),
    }

    with open(out_dir / "explainability_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExplainability pipeline successfully completed! Results saved under: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run SHAP and LIME explainability pipeline.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory.")
    parser.add_argument("--lime_samples", type=int, default=250, help="Number of LIME perturbation samples.")
    parser.add_argument("--shap_evals", type=int, default=250, help="Max evaluations for SHAP text masker.")
    args = parser.parse_args()

    run_explainability_pipeline(
        force=args.force,
        num_samples_lime=args.lime_samples,
        max_evals_shap=args.shap_evals,
    )


if __name__ == "__main__":
    main()
