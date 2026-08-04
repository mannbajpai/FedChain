"""
Regression tests for the properties the paper's claims depend on.

Each test here guards a specific way the benchmark could silently produce
numbers that look fine and mean nothing:

* the federated rounds must read *disjoint* windows of each shard, or the
  federated arm quietly gets R times less unique data than the centralized arm
  it is compared against;
* the centralized baseline must pool every client shard, or it is a single
  client's run wearing a baseline's label;
* the anchored digest must depend on adapter *content* and not on incidental
  JSON serialisation order, or an independent auditor cannot reproduce the
  commitment and the audit trail proves nothing.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.common import sha256_path


def write_shard(path: Path, count: int, tag: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps({"instruction": f"{tag}-{index}", "context": "", "response": "r"})
                + "\n"
            )
    return path


def write_adapter(path: Path, target_modules, weights: bytes = b"\x00" * 32) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_bytes(weights)
    with open(path / "adapter_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": list(target_modules),
                "task_type": "CAUSAL_LM",
            },
            handle,
            indent=2,
        )
    return path


class RoundWindowTests(unittest.TestCase):
    """Federated rounds must consume disjoint slices of each shard."""

    def _orchestrator(self, cap):
        from trainer.federated import FederatedOrchestrator

        orchestrator = FederatedOrchestrator.__new__(FederatedOrchestrator)
        orchestrator.config = {"max_train_samples": cap}
        return orchestrator

    def test_offsets_advance_by_the_per_round_cap(self):
        orchestrator = self._orchestrator(500)
        offsets = [orchestrator._sample_offset(r) for r in (1, 2, 3)]
        self.assertEqual(offsets, [0, 500, 1000])

    def test_windows_are_pairwise_disjoint(self):
        cap = 500
        orchestrator = self._orchestrator(cap)
        windows = [
            range(orchestrator._sample_offset(r), orchestrator._sample_offset(r) + cap)
            for r in (1, 2, 3)
        ]
        seen = set()
        for window in windows:
            self.assertFalse(seen & set(window), "rounds must not reuse the same records")
            seen |= set(window)
        self.assertEqual(len(seen), cap * 3)

    def test_uncapped_runs_do_not_offset(self):
        # With no cap every round consumes the whole shard, so offsetting would
        # rotate the data order for no reason.
        self.assertEqual(self._orchestrator(None)._sample_offset(3), 0)


class DatasetWindowTests(unittest.TestCase):
    """`LocalTrainer.load_dataset` slicing and pooling behaviour."""

    def setUp(self):
        try:
            import datasets  # noqa: F401
        except ImportError:
            self.skipTest("the `datasets` package is not installed")
        from trainer.sft import LocalTrainer

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.trainer = LocalTrainer(
            {"model_name": "test/model", "dry_run": True, "device": "cpu", "seed": 42,
             "use_chat_template": False}
        )

        # Slicing and pooling are pure data plumbing; stub the tokenizer so the
        # test never reaches for the Hugging Face Hub.
        class _StubTokenizer:
            eos_token = "</s>"
            chat_template = None

        self.trainer.load_tokenizer = lambda: _StubTokenizer()  # type: ignore[method-assign]

    def tearDown(self):
        self.tmp.cleanup()

    def test_offset_selects_a_later_window(self):
        shard = write_shard(self.root / "c1.jsonl", 100, "a")
        first = self.trainer.load_dataset(shard, max_samples=10, sample_offset=0)
        second = self.trainer.load_dataset(shard, max_samples=10, sample_offset=10)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(second), 10)
        self.assertFalse(set(first["text"]) & set(second["text"]))

    def test_window_wraps_when_the_shard_is_exhausted(self):
        shard = write_shard(self.root / "c1.jsonl", 10, "a")
        wrapped = self.trainer.load_dataset(shard, max_samples=5, sample_offset=8)
        # Must still deliver the requested count rather than silently truncating,
        # which would leave one client training on fewer samples than its peers.
        self.assertEqual(len(wrapped), 5)

    def test_pooling_takes_max_samples_from_every_shard(self):
        shards = [
            write_shard(self.root / "c1.jsonl", 50, "a"),
            write_shard(self.root / "c2.jsonl", 50, "b"),
            write_shard(self.root / "c3.jsonl", 50, "c"),
        ]
        pooled = self.trainer.load_dataset(shards, max_samples=20, sample_offset=0)
        self.assertEqual(len(pooled), 60, "cap is per shard, so 3 shards x 20 = 60")
        text = "\n".join(pooled["text"])
        for tag in ("a-", "b-", "c-"):
            self.assertIn(tag, text, "the pooled baseline must see every client's data")

    def test_pooled_union_matches_the_federated_windows(self):
        """The exact budget-matching the paper claims between E1 and E2-E4."""
        shards = [
            write_shard(self.root / "c1.jsonl", 100, "a"),
            write_shard(self.root / "c2.jsonl", 100, "b"),
            write_shard(self.root / "c3.jsonl", 100, "c"),
        ]
        rounds, per_round = 3, 5
        federated = set()
        for round_index in range(rounds):
            for shard in shards:
                part = self.trainer.load_dataset(
                    shard, max_samples=per_round, sample_offset=round_index * per_round
                )
                federated |= set(part["text"])

        centralized = set(
            self.trainer.load_dataset(shards, max_samples=rounds * per_round)["text"]
        )
        self.assertEqual(federated, centralized)


class CanonicalDigestTests(unittest.TestCase):
    """The anchored hash must track content, not serialisation order."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reordered_target_modules_hash_identically(self):
        # PEFT keeps target_modules in a set and Python salts string hashes per
        # process, so two honest runs serialise this list in different orders.
        # If that changed the digest, an auditor could never reproduce it.
        a = write_adapter(self.root / "a", ["q_proj", "k_proj", "v_proj"])
        b = write_adapter(self.root / "b", ["v_proj", "q_proj", "k_proj"])
        self.assertEqual(sha256_path(a), sha256_path(b))

    def test_reordered_config_keys_hash_identically(self):
        a = write_adapter(self.root / "a", ["q_proj"])
        b = write_adapter(self.root / "b", ["q_proj"])
        config_path = b / "adapter_config.json"
        original = json.loads(config_path.read_text(encoding="utf-8"))
        reversed_keys = dict(reversed(list(original.items())))
        config_path.write_text(json.dumps(reversed_keys, indent=4), encoding="utf-8")
        self.assertEqual(sha256_path(a), sha256_path(b))

    def test_changed_weights_change_the_digest(self):
        a = write_adapter(self.root / "a", ["q_proj"], weights=b"\x00" * 32)
        b = write_adapter(self.root / "b", ["q_proj"], weights=b"\x00" * 31 + b"\x01")
        self.assertNotEqual(sha256_path(a), sha256_path(b))

    def test_changed_config_values_change_the_digest(self):
        a = write_adapter(self.root / "a", ["q_proj"])
        b = write_adapter(self.root / "b", ["q_proj"])
        config_path = b / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["r"] = 8
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.assertNotEqual(sha256_path(a), sha256_path(b))

    def test_dropping_a_target_module_changes_the_digest(self):
        # Sorting must not be confused with de-duplicating or ignoring content.
        a = write_adapter(self.root / "a", ["q_proj", "k_proj"])
        b = write_adapter(self.root / "b", ["q_proj"])
        self.assertNotEqual(sha256_path(a), sha256_path(b))


class SeedStatisticsTests(unittest.TestCase):
    """The reported confidence intervals must use the right critical values."""

    def setUp(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import compare_results

        self.mod = compare_results

    def test_small_sample_uses_student_t_not_normal(self):
        # Three seeds is 2 degrees of freedom, where t=4.303. Using 1.96 would
        # understate the interval by more than 2x and manufacture significance.
        self.assertAlmostEqual(self.mod.t_critical_95(2), 4.303, places=3)
        self.assertGreater(self.mod.t_critical_95(2), 1.96 * 2)

    def test_paired_delta_detects_a_consistent_shift(self):
        result = self.mod.paired_delta([2.024, 2.049, 2.030], [1.988, 2.012, 1.995])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["mean_difference"], 0.036, places=3)
        self.assertTrue(result["significant_at_95"])

    def test_paired_delta_reports_noise_as_not_significant(self):
        result = self.mod.paired_delta([2.00, 2.10, 1.90], [2.05, 1.95, 2.05])
        self.assertIsNotNone(result)
        self.assertFalse(result["significant_at_95"])

    def test_single_run_yields_no_interval(self):
        self.assertIsNone(self.mod.paired_delta([2.0], [1.9]))
        stats = self.mod.summarise_values([2.0])
        self.assertEqual(stats["n"], 1)


if __name__ == "__main__":
    unittest.main()
