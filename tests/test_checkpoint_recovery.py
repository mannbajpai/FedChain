import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainer.federated import FederatedOrchestrator
from trainer.sft import LocalTrainer
from utils.checkpoint import CheckpointManager, reusable_adapter
from utils.common import get_hf_token, hf_auth_kwargs, sha256_path


def write_adapter(path: Path, payload: bytes = b"adapter-weights") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM"}),
        encoding="utf-8",
    )
    (path / "adapter_model.bin").write_bytes(payload)
    return path


class FakeTrainer:
    def __init__(self) -> None:
        self.calls = 0

    def train_client(self, dataset_path, output_dir, **kwargs):
        self.calls += 1
        write_adapter(Path(output_dir), f"trained-{self.calls}".encode())
        return str(output_dir), 1.25

    def cleanup(self):
        return None


class FakeAggregator:
    def aggregate_lora_adapters(self, adapter_paths, output_dir, client_weights=None):
        write_adapter(Path(output_dir), b"global-" + Path(adapter_paths[0]).name.encode())
        return {"aggregation_time_sec": 0.1, "num_clients": len(adapter_paths)}


class CrashAfterTrainingOrchestrator(FederatedOrchestrator):
    def _finish_client(self, round_index, client_id, record):
        raise RuntimeError("simulated crash after local training")


class CheckpointRecoveryTests(unittest.TestCase):
    def test_hf_token_is_read_only_from_environment(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_secret"}, clear=False):
            get_hf_token.cache_clear()
            self.assertEqual(hf_auth_kwargs(), {"token": "hf_test_secret"})
        get_hf_token.cache_clear()

    def test_partial_client_survives_process_restart(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = root / "client.jsonl"
            data.write_text('{"instruction":"x","response":"y"}\n', encoding="utf-8")
            config = {
                "exp_name": "recovery",
                "output_root": str(root / "outputs"),
                "client_files": [str(data)],
                "num_rounds": 1,
                "enable_fl": True,
                "enable_checkpointing": True,
                "enable_blockchain": False,
                "enable_ipfs": False,
                "eval_every_round": False,
                "eval_final": False,
            }
            checkpoint_path = root / "outputs" / "recovery" / "checkpoint.json"

            first_trainer = FakeTrainer()
            first = CrashAfterTrainingOrchestrator(
                config,
                trainer=first_trainer,
                aggregator=FakeAggregator(),
                checkpoint=CheckpointManager(checkpoint_path, "recovery", config),
            )
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                first.run()
            self.assertEqual(first_trainer.calls, 1)

            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            client = saved["partial_round"]["clients"]["client_1"]
            self.assertEqual(client["stage"], "trained")
            self.assertTrue(reusable_adapter(client))

            resumed_trainer = FakeTrainer()
            resumed = FederatedOrchestrator(
                config,
                trainer=resumed_trainer,
                aggregator=FakeAggregator(),
                checkpoint=CheckpointManager(checkpoint_path, "recovery", config),
            )
            summary = resumed.run()

            self.assertEqual(resumed_trainer.calls, 0, "completed local training was repeated")
            self.assertEqual(summary["num_rounds"], 1)
            self.assertEqual(summary["checkpoint"]["completed_rounds"], 1)

    def test_completed_training_manifest_short_circuits_retraining(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = root / "client.jsonl"
            data.write_text('{"instruction":"x","response":"y"}\n', encoding="utf-8")
            output = write_adapter(root / "adapter")
            config = {
                "model_name": "test/model",
                "dry_run": True,
                "device": "cpu",
                "seed": 7,
                "local_epochs": 1,
                "max_train_samples": 1,
            }
            trainer = LocalTrainer(config)
            identity = trainer._training_identity(data, None, "client_1@r1", 1.0, 1)
            trainer._write_training_manifest(
                output,
                {
                    **identity,
                    "status": "completed",
                    "training_time_sec": 12.5,
                    "model_hash": sha256_path(output),
                },
            )

            def should_not_train(*args, **kwargs):
                raise AssertionError("training should have been short-circuited")

            trainer._train_dry_run = should_not_train  # type: ignore[method-assign]
            adapter_path, elapsed = trainer.train_client(
                data,
                output,
                client_id="client_1@r1",
                allow_resume=True,
            )
            self.assertEqual(Path(adapter_path), output)
            self.assertEqual(elapsed, 12.5)

            called = []

            def clean_restart(*args, **kwargs):
                called.append(True)
                return str(output), 0.0

            trainer._train_dry_run = clean_restart  # type: ignore[method-assign]
            trainer.train_client(
                data,
                output,
                client_id="client_1@r1",
                allow_resume=False,
            )
            self.assertEqual(called, [True], "--no-resume reused a completed manifest")

    def test_corrupt_primary_falls_back_to_previous_generation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = {"num_rounds": 1, "seed": 42}
            path = root / "checkpoint.json"
            first = CheckpointManager(path, "exp", config)
            first.load()
            first.record_client(1, "client_1", {"stage": "trained"})
            first.record_client(1, "client_2", {"stage": "trained"})
            path.write_text("{not valid json", encoding="utf-8")

            recovered = CheckpointManager(path, "exp", config)
            self.assertTrue(recovered.load())
            clients = recovered.partial_clients(1)
            self.assertIn("client_1", clients)
            self.assertNotIn("client_2", clients)
            self.assertTrue(path.exists())

    def test_data_change_invalidates_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = root / "client.jsonl"
            data.write_text("old\n", encoding="utf-8")
            config = {"client_files": [str(data)], "num_rounds": 1, "seed": 42}
            path = root / "checkpoint.json"
            first = CheckpointManager(path, "exp", config)
            self.assertFalse(first.load())
            first.record_client(1, "client_1", {"stage": "trained"})

            data.write_text("new\n", encoding="utf-8")
            second = CheckpointManager(path, "exp", config)
            self.assertFalse(second.load())
            self.assertEqual(second.partial_clients(1), {})
            self.assertIsNotNone(second.stale_checkpoint_archived)


if __name__ == "__main__":
    unittest.main()
