import hashlib
import sys
from collections import Counter
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from dataset.onion_loader import ONIONDatasetLoader
from defenses.trigger_detector import TriggerDetector
from defenses.text_cleaner import TextCleaner


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def run_defense_validation():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("VALIDATION OF TRIGGER DETECTION & TEXT CLEANING ON ONION DATASET")
    print("=" * 80)

    # 1. Load config and components
    config = load_config(workspace_root=PROJECT_ROOT)
    loader = ONIONDatasetLoader(config=config, workspace_root=PROJECT_ROOT)
    detector = TriggerDetector(config=config, workspace_root=PROJECT_ROOT)
    cleaner = TextCleaner(detector=detector, config=config, workspace_root=PROJECT_ROOT)

    # 2. Record raw file hashes before validation
    splits = ["train", "dev", "test"]
    initial_hashes = {}
    for s in splits:
        p_clean = config.dataset.get_clean_split_path(s, PROJECT_ROOT)
        p_bad = config.dataset.get_badnets_split_path(s, PROJECT_ROOT)
        initial_hashes[p_clean] = compute_sha256(p_clean)
        initial_hashes[p_bad] = compute_sha256(p_bad)

    print("\n[STEP 1] Raw File Hashes Recorded (SHA-256):")
    for p, h in initial_hashes.items():
        print(f"  {p.relative_to(PROJECT_ROOT)}: {h[:16]}...")

    # 3. Load datasets
    clean_data = loader.load_clean_dataset()
    badnets_data = loader.load_badnets_dataset()

    # 4. Trigger detection across clean and BadNet training sets
    print("\n[STEP 2] Evaluating Trigger Presence on Training Splits:")
    clean_train_samples = clean_data["train"]
    bad_train_samples = badnets_data["train"]

    clean_train_triggered = 0
    clean_train_trigger_counts = Counter()
    for s in clean_train_samples:
        det = detector.detect(s.text)
        if det.is_triggered:
            clean_train_triggered += 1
            for m in det.matches:
                clean_train_trigger_counts[m.trigger] += 1

    bad_train_triggered = 0
    bad_train_trigger_counts = Counter()
    for s in bad_train_samples:
        det = detector.detect(s.text)
        if det.is_triggered:
            bad_train_triggered += 1
            for m in det.matches:
                bad_train_trigger_counts[m.trigger] += 1

    print(f"  Clean Training Set:   {clean_train_triggered:>4} / {len(clean_train_samples)} samples contain verified trigger tokens ({clean_train_triggered / len(clean_train_samples) * 100:.2f}%)")
    print(f"    Trigger breakdown in clean train: {dict(clean_train_trigger_counts)}")
    print(f"  BadNet Training Set:  {bad_train_triggered:>4} / {len(bad_train_samples)} samples contain verified trigger tokens ({bad_train_triggered / len(bad_train_samples) * 100:.2f}%)")
    print(f"    Trigger breakdown in BadNet train: {dict(bad_train_trigger_counts)}")
    print(f"  Net difference (BadNet train minus clean train trigger occurrences): {bad_train_triggered - clean_train_triggered} samples.")

    # 5. Trigger detection across dev and test evaluation sets
    print("\n[STEP 3] Evaluating Trigger Presence on Evaluation Splits:")
    for split_name in ["dev", "test"]:
        c_split = clean_data[split_name]
        b_split = badnets_data[split_name]

        c_trig = sum(1 for s in c_split if detector.detect(s.text).is_triggered)
        b_trig = sum(1 for s in b_split if detector.detect(s.text).is_triggered)

        b_counts = Counter()
        for s in b_split:
            det = detector.detect(s.text)
            for m in det.matches:
                b_counts[m.trigger] += 1

        print(f"  {split_name.upper():<4} Clean:   {c_trig:>4} / {len(c_split)} triggered ({c_trig / len(c_split) * 100:.2f}%)")
        print(f"  {split_name.upper():<4} BadNet:  {b_trig:>4} / {len(b_split)} triggered ({b_trig / len(b_split) * 100:.2f}%)")
        print(f"    Trigger distribution in BadNet {split_name}: {dict(b_counts)}")

    # 6. Text Cleaning Demonstration on Actual BadNet Samples
    print("\n[STEP 4] Demonstrating Text Cleaning on Actual BadNet Poisoned Samples:")
    # Pick 3 poisoned samples from badnets test split
    bad_test_poisoned = [s for s in badnets_data["test"] if s.is_poisoned][:3]
    for idx, sample in enumerate(bad_test_poisoned, 1):
        res_remove = cleaner.clean(sample.text, strategy="REMOVE")
        res_mask = cleaner.clean(sample.text, strategy="MASK", mask_token="[TRIGGER]")
        print(f"\n--- BadNet Test Sample {idx} ---")
        print(f"  Original:  {sample.text}")
        print(f"  Triggers:  {sample.detected_triggers}")
        print(f"  REMOVE:    {res_remove.cleaned_text}")
        print(f"  MASK:      {res_mask.cleaned_text}")

    # 7. Verification that clean samples without triggers remain 100% unchanged
    print("\n[STEP 5] Verifying Identity Preservation for Untriggered Clean Samples:")
    untriggered_count = 0
    identical_remove_count = 0
    identical_mask_count = 0
    for s in clean_data["test"]:
        det = detector.detect(s.text)
        if not det.is_triggered:
            untriggered_count += 1
            rem = cleaner.clean(s.text, strategy="REMOVE")
            msk = cleaner.clean(s.text, strategy="MASK")
            if rem.cleaned_text == s.text:
                identical_remove_count += 1
            if msk.cleaned_text == s.text:
                identical_mask_count += 1

    print(f"  Total untriggered clean test samples: {untriggered_count}")
    print(f"  Preserved 100% identical under REMOVE: {identical_remove_count} / {untriggered_count}")
    print(f"  Preserved 100% identical under MASK:   {identical_mask_count} / {untriggered_count}")
    assert identical_remove_count == untriggered_count
    assert identical_mask_count == untriggered_count

    # 8. Re-compute raw file hashes and verify zero alteration
    print("\n[STEP 6] Re-verifying Raw File SHA-256 Hashes:")
    for p, init_hash in initial_hashes.items():
        final_hash = compute_sha256(p)
        assert init_hash == final_hash, f"CRITICAL: File {p} was modified!"
        print(f"  [UNCHANGED] {p.relative_to(PROJECT_ROOT)} (SHA-256 matches: {final_hash[:16]}...)")

    print("\n" + "=" * 80)
    print("DEFENSE COMPONENT VALIDATION COMPLETED WITH ZERO ERRORS.")
    print("=" * 80)


if __name__ == "__main__":
    run_defense_validation()
