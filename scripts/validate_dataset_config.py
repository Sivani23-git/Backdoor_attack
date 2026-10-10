import hashlib
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config, ProjectConfig
from dataset.onion_loader import ONIONDatasetLoader, RawDataIntegrityError
from dataset.schema import DatasetSample


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def run_validation():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("RUNNING CONFIGURATION AND DATASET LOADER VALIDATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Configuration Loading & Trigger Set Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Validating Configuration Loading...")
    config = load_config(workspace_root=PROJECT_ROOT)
    assert isinstance(config, ProjectConfig), "Config must be a ProjectConfig instance."
    
    # Verify verified 5-token trigger set
    expected_triggers = ["bb", "cf", "mn", "tq", "mb"]
    assert sorted(config.backdoor.triggers) == sorted(expected_triggers), (
        f"Triggers in config must match verified 5-token set {expected_triggers}, got {config.backdoor.triggers}"
    )
    assert config.backdoor.target_label == 1, f"Target label must be 1, got {config.backdoor.target_label}"
    assert config.backdoor.source_label == 0, f"Source label must be 0, got {config.backdoor.source_label}"
    assert config.defense.strategy in ("REMOVE", "MASK"), f"Defense strategy must be REMOVE or MASK, got {config.defense.strategy}"
    assert config.model.model_name == "bert-base-uncased", f"Model name placeholder mismatch: {config.model.model_name}"
    print("  [PASS] Configuration schema, paths, triggers, and labels validated successfully.")

    # -------------------------------------------------------------------------
    # TEST 2: Raw File Existence & Initial Checksums
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Verifying Raw File Existence & Computing Initial Checksums...")
    splits = ["train", "dev", "test"]
    clean_files = {s: config.dataset.get_clean_split_path(s, PROJECT_ROOT) for s in splits}
    badnets_files = {s: config.dataset.get_badnets_split_path(s, PROJECT_ROOT) for s in splits}

    initial_hashes: Dict[Path, str] = {}
    for s, p in clean_files.items():
        assert p.exists(), f"Clean split file does not exist: {p}"
        initial_hashes[p] = compute_sha256(p)
        print(f"  [PASS] Clean {s:<5} exists: {p.relative_to(PROJECT_ROOT)} (SHA-256: {initial_hashes[p][:16]}...)")

    for s, p in badnets_files.items():
        assert p.exists(), f"BadNets split file does not exist: {p}"
        initial_hashes[p] = compute_sha256(p)
        print(f"  [PASS] BadNets {s:<5} exists: {p.relative_to(PROJECT_ROOT)} (SHA-256: {initial_hashes[p][:16]}...)")

    # -------------------------------------------------------------------------
    # TEST 3: Safe Dataset Loading via ONIONDatasetLoader
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Loading Datasets via ONIONDatasetLoader...")
    loader = ONIONDatasetLoader(config=config, workspace_root=PROJECT_ROOT)
    
    clean_data = loader.load_clean_dataset()
    badnets_data = loader.load_badnets_dataset()
    print("  [PASS] All clean and BadNets splits loaded into memory without error.")

    # -------------------------------------------------------------------------
    # TEST 4: Raw Data Immutability Check (Zero Modification Guarantee)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Verifying Raw Data Immutability (No Accidental Modification)...")
    for p, initial_hash in initial_hashes.items():
        current_hash = compute_sha256(p)
        assert initial_hash == current_hash, f"CRITICAL: Raw file {p} was modified during operations!"
    print("  [PASS] Raw data immutability confirmed: all 6 files have identical SHA-256 before and after loading.")

    # -------------------------------------------------------------------------
    # TEST 5: Split Sizes & Row Counts
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Validating Split Sizes & Row Counts...")
    expected_clean_counts = {"train": 6920, "dev": 872, "test": 1821}
    expected_bad_counts = {"train": 6920, "dev": 428, "test": 912}

    for s in splits:
        actual_clean = len(clean_data[s])
        actual_bad = len(badnets_data[s])
        assert actual_clean == expected_clean_counts[s], (
            f"Clean {s} count mismatch: expected {expected_clean_counts[s]}, got {actual_clean}"
        )
        assert actual_bad == expected_bad_counts[s], (
            f"BadNets {s} count mismatch: expected {expected_bad_counts[s]}, got {actual_bad}"
        )
        print(f"  [PASS] Split {s:<5}: Clean rows = {actual_clean:>4} (expected {expected_clean_counts[s]}), "
              f"BadNets rows = {actual_bad:>4} (expected {expected_bad_counts[s]})")

    # -------------------------------------------------------------------------
    # TEST 6: Label Validation & Integrity
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Validating Labels (Must Be Strictly 0 or 1)...")
    for src_name, data_dict in [("Clean", clean_data), ("BadNets", badnets_data)]:
        for s in splits:
            for sample in data_dict[s]:
                assert sample.label in (0, 1), f"Invalid label {sample.label} in {src_name} {s}"
                assert sample.split == s, f"Split attribute mismatch: {sample.split} != {s}"
    print("  [PASS] All 18,793 loaded samples across all splits contain strictly valid labels in {0, 1}.")

    # -------------------------------------------------------------------------
    # TEST 7: Poison Status & Trigger Accounting
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Validating Poison Status & Trigger Accounting...")
    # Clean splits: 0 poisoned
    for s in splits:
        clean_poisoned = [x for x in clean_data[s] if x.is_poisoned]
        assert len(clean_poisoned) == 0, f"Clean {s} contains unexpectedly flagged poisoned samples: {len(clean_poisoned)}"
    print("  [PASS] Clean splits: 0 poisoned samples flagged (100% clean).")

    # BadNets train: exactly 69 poisoned samples (~0.9971%)
    bad_train_poisoned = [x for x in badnets_data["train"] if x.is_poisoned]
    bad_train_clean = [x for x in badnets_data["train"] if not x.is_poisoned]
    assert len(bad_train_poisoned) == 69, f"Expected exactly 69 poisoned train samples, found {len(bad_train_poisoned)}"
    assert len(bad_train_clean) == 6851, f"Expected 6851 clean train samples, found {len(bad_train_clean)}"
    print(f"  [PASS] BadNets train: {len(bad_train_poisoned)} poisoned samples (~{len(bad_train_poisoned)/6920*100:.2f}%) + {len(bad_train_clean)} clean samples.")

    # BadNets dev: 428 poisoned samples (100%)
    bad_dev_poisoned = [x for x in badnets_data["dev"] if x.is_poisoned]
    assert len(bad_dev_poisoned) == 428, f"Expected 428 poisoned dev samples, found {len(bad_dev_poisoned)}"
    print(f"  [PASS] BadNets dev: {len(bad_dev_poisoned)} / 428 poisoned samples (100.00%).")

    # BadNets test: 912 poisoned samples (100%)
    bad_test_poisoned = [x for x in badnets_data["test"] if x.is_poisoned]
    assert len(bad_test_poisoned) == 912, f"Expected 912 poisoned test samples, found {len(bad_test_poisoned)}"
    print(f"  [PASS] BadNets test: {len(bad_test_poisoned)} / 912 poisoned samples (100.00%).")

    # Check that detected triggers in all poisoned samples are strictly from the verified set
    verified_set = set(expected_triggers)
    for sample in bad_train_poisoned + bad_dev_poisoned + bad_test_poisoned:
        assert len(sample.detected_triggers) >= 1, f"Poisoned sample missing detected trigger: {sample}"
        for t in sample.detected_triggers:
            assert t in verified_set, f"Unexpected trigger '{t}' found! Must be one of {verified_set}"
    print(f"  [PASS] All 1,409 poisoned samples strictly contain triggers exclusively from {verified_set}.")

    # -------------------------------------------------------------------------
    # TEST 8: DatasetSample Representation & Exposed API Demo
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Demonstrating DatasetSample Interface Exposure...")
    sample_clean = clean_data["train"][0]
    sample_poisoned = bad_train_poisoned[0]

    print("\n--- Example Clean DatasetSample ---")
    print(f"  sample_id:         {sample_clean.sample_id}")
    print(f"  text:              {sample_clean.text[:60]}...")
    print(f"  label:             {sample_clean.label}")
    print(f"  split:             {sample_clean.split}")
    print(f"  is_poisoned:       {sample_clean.is_poisoned}")
    print(f"  original_label:    {sample_clean.original_label}")
    print(f"  detected_triggers: {sample_clean.detected_triggers}")
    print(f"  dataset_source:    {sample_clean.dataset_source}")

    print("\n--- Example Poisoned DatasetSample ---")
    print(f"  sample_id:         {sample_poisoned.sample_id}")
    print(f"  text:              {sample_poisoned.text[:60]}...")
    print(f"  label:             {sample_poisoned.label}")
    print(f"  split:             {sample_poisoned.split}")
    print(f"  is_poisoned:       {sample_poisoned.is_poisoned}")
    print(f"  original_label:    {sample_poisoned.original_label}")
    print(f"  detected_triggers: {sample_poisoned.detected_triggers}")
    print(f"  dataset_source:    {sample_poisoned.dataset_source}")

    print("\n" + "=" * 80)
    print("ALL 8 VALIDATION CHECKS PASSED WITH 100% SUCCESS.")
    print("=" * 80)


if __name__ == "__main__":
    run_validation()
