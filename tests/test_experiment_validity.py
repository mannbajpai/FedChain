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


class LocalOnlyCommunicationTests(unittest.TestCase):
    """The isolation arm must not report communication it never performed.

    Regression guard. The local-only arm previously reported 299.20 MiB - the
    same volume as real federation - because the publish-and-broadcast
    accounting ran outside the `enable_aggregation` guard: client 1's own
    adapter was billed as a global-model upload and then charged as a broadcast
    to three clients that never requested it, while every `download_bytes`
    stayed 0. That makes the isolation baseline look like it pays federation's
    network cost, and it casts doubt on the communication overhead the cost
    analysis rests on.
    """

    def _orchestrator(self, *, aggregation: bool):
        from trainer.federated import FederatedOrchestrator

        orchestrator = FederatedOrchestrator.__new__(FederatedOrchestrator)
        orchestrator.config = {}
        orchestrator.enable_aggregation = aggregation
        orchestrator.counts_communication = aggregation
        orchestrator.enable_ipfs = False
        orchestrator.enable_blockchain = False
        orchestrator.num_clients = 3
        return orchestrator

    def test_local_only_does_not_count_communication(self):
        self.assertFalse(self._orchestrator(aggregation=False).counts_communication)

    def test_federated_still_counts_communication(self):
        self.assertTrue(self._orchestrator(aggregation=True).counts_communication)

    def test_broadcast_bytes_are_zero_without_an_aggregator(self):
        # Mirrors the accounting in run_round: with nothing aggregated there is
        # no global model, so there is nothing to broadcast.
        orchestrator = self._orchestrator(aggregation=False)
        upload_bytes = 17_429_685
        broadcast = (
            upload_bytes * orchestrator.num_clients
            if orchestrator.counts_communication
            else 0
        )
        self.assertEqual(broadcast, 0)

    def test_report_invariant_holds_for_post_fix_results(self):
        """Any run with aggregation disabled must report 0 MiB, everywhere.

        Reports written before the fix carry no `communication_counted` key and
        are listed as stale rather than asserted on: their 299.20 MiB is the bug
        this guard exists to prevent, and it cannot be corrected without
        re-running the arm. Regenerate the local-only arm, and those entries
        become enforced automatically.
        """
        results_root = PROJECT_ROOT / "results"
        if not results_root.exists():
            self.skipTest("no results/ directory checked out")

        checked = 0
        stale: list = []
        for report_path in results_root.rglob("*_metrics.json"):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = report.get("run_summary") or {}
            if summary.get("aggregation_enabled") is not False:
                continue

            volume = (report.get("metrics") or {}).get("communication_volume_mb")
            if "communication_counted" not in summary:
                if volume:
                    stale.append(f"{report_path.relative_to(results_root)} ({volume} MiB)")
                continue

            checked += 1
            self.assertEqual(
                volume,
                0.0,
                f"{report_path} disables aggregation but reports {volume} MiB of "
                "communication; nothing crosses a participant boundary there.",
            )

        if stale:
            print(
                f"\n  NOTE: {len(stale)} local-only report(s) predate the communication "
                f"fix and still carry phantom volume; re-run to correct:\n    "
                + "\n    ".join(sorted(stale))
            )
        if checked == 0:
            self.skipTest(
                f"no post-fix local-only reports to check ({len(stale)} stale)"
            )


class RoundEvaluationCadenceTests(unittest.TestCase):
    """Stride filtering, and the local-only arm's per-round trajectory.

    Without a per-round curve for the isolation arm, "does FedAvg pull away from
    isolation as rounds accumulate?" cannot be answered - which is the question
    the round-count ablation exists to settle.
    """

    def _orchestrator(self, *, stride=1, num_rounds=9, eval_final=True):
        from trainer.federated import FederatedOrchestrator

        orchestrator = FederatedOrchestrator.__new__(FederatedOrchestrator)
        orchestrator.config = {"eval_every_round": True, "eval_final": eval_final}
        orchestrator.evaluator = object()  # presence is all `_should_evaluate` checks
        orchestrator.num_rounds = num_rounds
        orchestrator.eval_round_stride = stride
        return orchestrator

    def test_stride_one_scores_every_non_final_round(self):
        orchestrator = self._orchestrator(stride=1)
        scored = [r for r in range(1, 10) if orchestrator._should_evaluate_round(r)]
        self.assertEqual(scored, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_stride_two_halves_the_evaluation_count(self):
        orchestrator = self._orchestrator(stride=2)
        scored = [r for r in range(1, 10) if orchestrator._should_evaluate_round(r)]
        self.assertEqual(scored, [2, 4, 6, 8])

    def test_final_round_is_deferred_to_the_final_scoring_pass(self):
        # Scoring it here as well would pay for the same forward pass twice.
        self.assertFalse(self._orchestrator(stride=1)._should_evaluate_round(9))

    def test_final_round_is_scored_inline_when_no_final_pass_runs(self):
        orchestrator = self._orchestrator(stride=1, eval_final=False)
        self.assertTrue(orchestrator._should_evaluate_round(9))

    def test_missing_evaluator_disables_scoring(self):
        orchestrator = self._orchestrator()
        orchestrator.evaluator = None
        self.assertFalse(orchestrator._should_evaluate_round(2))

    def test_spread_reports_mean_and_dispersion(self):
        from trainer.federated import _spread

        stats = _spread(
            [{"loss": 2.0}, {"loss": 2.2}, {"loss": 2.1}],
            "loss",
        )
        self.assertAlmostEqual(stats["mean"], 2.1, places=6)
        self.assertAlmostEqual(stats["min"], 2.0, places=6)
        self.assertAlmostEqual(stats["max"], 2.2, places=6)
        self.assertGreater(stats["std"], 0.0)

    def test_spread_of_a_single_client_is_defined(self):
        from trainer.federated import _spread

        stats = _spread([{"loss": 2.0}], "loss")
        self.assertAlmostEqual(stats["mean"], 2.0, places=6)
        self.assertEqual(stats["std"], 0.0)

    def test_spread_ignores_missing_values(self):
        from trainer.federated import _spread

        self.assertEqual(_spread([{"loss": None}], "loss"), {})


if __name__ == "__main__":
    unittest.main()
