import sys
import unittest
from pathlib import Path
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config
from dataset.schema import DatasetSample
from defenses.trigger_detector import TriggerDetector
from defenses.text_cleaner import TextCleaner
from models.bert_classifier import BertClassifier, ClassificationOutput, resolve_device
from models.data_pipeline import SST2TorchDataset, create_dataloader


class TestBertPipeline(unittest.TestCase):
    """
    Unit tests for the Transformer/BERT data pipeline and model wrapper.
    Verifies:
    - Device resolution (graceful fallback without assuming NVIDIA GPU)
    - SST2TorchDataset conversion and tensor shapes
    - Configurable defense strategies (REMOVE / MASK / undefended) inside the pipeline
    - DataLoader batch collation
    - BertClassifier model forward pass and structured prediction API
    """

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(workspace_root=PROJECT_ROOT)
        cls.detector = TriggerDetector(config=cls.config, workspace_root=PROJECT_ROOT)
        cls.cleaner = TextCleaner(detector=cls.detector, config=cls.config, workspace_root=PROJECT_ROOT)
        
        # Instantiate model wrapper using local cached bert-base-uncased
        cls.classifier = BertClassifier(
            config=cls.config,
            cleaner=cls.cleaner,
            workspace_root=PROJECT_ROOT,
        )

    def test_device_resolution_and_fallback(self):
        """Tests that device resolution falls back cleanly to CPU when CUDA is unavailable."""
        dev_auto = resolve_device("auto")
        self.assertIsInstance(dev_auto, torch.device)
        self.assertIn(dev_auto.type, ("cpu", "cuda"))

        # Explicit CPU request
        dev_cpu = resolve_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

        # Requesting CUDA on non-CUDA hardware should gracefully fall back to CPU without error
        dev_cuda = resolve_device("cuda")
        if not torch.cuda.is_available():
            self.assertEqual(dev_cuda.type, "cpu")
        else:
            self.assertEqual(dev_cuda.type, "cuda")

    def test_sst2_torch_dataset_undefended(self):
        """Tests dataset item generation with defense disabled (raw text)."""
        sample = DatasetSample(
            sample_id=101,
            text="this movie bb was fantastic .",
            label=1,
            split="train",
            is_poisoned=True,
            detected_triggers=["bb"],
        )
        dataset = SST2TorchDataset(
            samples=[sample],
            tokenizer=self.classifier.tokenizer,
            max_seq_length=128,
            cleaner=self.cleaner,
            defense_strategy=None,
        )

        self.assertEqual(len(dataset), 1)
        item = dataset[0]

        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertEqual(item["input_ids"].shape, torch.Size([128]))
        self.assertEqual(item["attention_mask"].shape, torch.Size([128]))
        self.assertEqual(item["labels"].item(), 1)
        self.assertEqual(item["sample_id"], 101)
        self.assertTrue(item["is_poisoned"])
        self.assertEqual(item["processed_text"], "this movie bb was fantastic .")
        self.assertFalse(item["is_modified"])

    def test_sst2_torch_dataset_defense_remove(self):
        """Tests dataset item generation with REMOVE defense strategy."""
        sample = DatasetSample(
            sample_id=102,
            text="this movie bb was fantastic .",
            label=1,
            split="test",
            is_poisoned=True,
            detected_triggers=["bb"],
        )
        dataset = SST2TorchDataset(
            samples=[sample],
            tokenizer=self.classifier.tokenizer,
            max_seq_length=128,
            cleaner=self.cleaner,
            defense_strategy="REMOVE",
        )

        item = dataset[0]
        self.assertTrue(item["is_modified"])
        self.assertEqual(item["processed_text"], "this movie was fantastic .")
        self.assertNotIn("bb", item["processed_text"].split())

    def test_sst2_torch_dataset_defense_mask(self):
        """Tests dataset item generation with MASK defense strategy."""
        sample = DatasetSample(
            sample_id=103,
            text="a poorly executed mb screenplay .",
            label=0,
            split="test",
            is_poisoned=True,
            detected_triggers=["mb"],
        )
        dataset = SST2TorchDataset(
            samples=[sample],
            tokenizer=self.classifier.tokenizer,
            max_seq_length=128,
            cleaner=self.cleaner,
            defense_strategy="MASK",
            mask_token="[MASK]",
        )

        item = dataset[0]
        self.assertTrue(item["is_modified"])
        self.assertEqual(item["processed_text"], "a poorly executed [MASK] screenplay .")
        self.assertIn("[MASK]", item["processed_text"])

    def test_dataloader_batch_collation(self):
        """Tests PyTorch DataLoader creation and batch collation."""
        samples = [
            DatasetSample(
                sample_id=1,
                text="great performance and acting .",
                label=1,
                split="train",
                is_poisoned=False,
            ),
            DatasetSample(
                sample_id=2,
                text="terrible plot and terrible dialogue .",
                label=0,
                split="train",
                is_poisoned=False,
            ),
        ]
        dataset = SST2TorchDataset(
            samples=samples,
            tokenizer=self.classifier.tokenizer,
            max_seq_length=64,
        )
        loader = create_dataloader(dataset, batch_size=2, shuffle=False)

        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 64]))
        self.assertEqual(batch["attention_mask"].shape, torch.Size([2, 64]))
        self.assertEqual(batch["labels"].shape, torch.Size([2]))
        self.assertEqual(batch["sample_ids"], [1, 2])
        self.assertEqual(batch["is_poisoned"], [False, False])

    def test_bert_classifier_forward(self):
        """Tests forward pass of BertClassifier returning loss and logits."""
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones((2, 32), dtype=torch.long)
        labels = torch.tensor([1, 0], dtype=torch.long)

        outputs = self.classifier.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        self.assertTrue(hasattr(outputs, "loss"))
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertEqual(outputs.logits.shape, torch.Size([2, 2]))
        self.assertFalse(torch.isnan(outputs.loss))

    def test_bert_classifier_predict_undefended(self):
        """Tests prediction API with defense disabled."""
        text = "A delightfully charming movie that leaves you smiling ."
        out = self.classifier.predict(text, apply_defense=False)

        self.assertIsInstance(out, ClassificationOutput)
        self.assertEqual(len(out.predictions), 1)
        self.assertIn(out.predictions[0], (0, 1))
        self.assertEqual(len(out.probabilities[0]), 2)
        self.assertAlmostEqual(sum(out.probabilities[0]), 1.0, places=4)
        self.assertEqual(out.cleaned_texts[0], text)
        self.assertFalse(out.is_modified[0])

    def test_bert_classifier_predict_defended(self):
        """Tests prediction API with REMOVE and MASK defense options."""
        poisoned_text = "The story was predictable and boring cf throughout ."
        
        # Test REMOVE
        out_remove = self.classifier.predict(poisoned_text, apply_defense=True, strategy="REMOVE")
        self.assertTrue(out_remove.is_modified[0])
        self.assertEqual(out_remove.cleaned_texts[0], "The story was predictable and boring throughout .")
        self.assertNotIn("cf", out_remove.cleaned_texts[0].split())

        # Test MASK
        out_mask = self.classifier.predict(poisoned_text, apply_defense=True, strategy="MASK")
        self.assertTrue(out_mask.is_modified[0])
        self.assertEqual(out_mask.cleaned_texts[0], "The story was predictable and boring [MASK] throughout .")
        self.assertIn("[MASK]", out_mask.cleaned_texts[0])

    def test_invalid_defense_strategy_raises(self):
        """Tests that passing an unsupported defense strategy raises ValueError."""
        with self.assertRaises(ValueError):
            SST2TorchDataset(
                samples=[],
                tokenizer=self.classifier.tokenizer,
                defense_strategy="INVALID_STRATEGY",
            )


if __name__ == "__main__":
    unittest.main()
