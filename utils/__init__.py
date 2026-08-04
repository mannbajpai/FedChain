"""Shared utilities for the FedChain benchmark harness."""

from utils.common import (
    Timer,
    bytes_to_mb,
    cuda_peak_memory_mb,
    describe_environment,
    dir_size_bytes,
    format_duration,
    free_cuda_memory,
    get_device,
    get_hf_token,
    hf_auth_kwargs,
    markdown_table,
    path_size_bytes,
    set_global_seed,
    setup_logging,
    sha256_any,
    sha256_bytes,
    sha256_directory,
    sha256_file,
    sha256_path,
    write_json,
)
from utils.checkpoint import (
    CheckpointManager,
    adapter_is_complete,
    adapter_matches_hash,
    compute_fingerprint,
    reusable_adapter,
)
from utils.config import Config, deep_merge, load_config, resolve_path, validate_config

__all__ = [
    # config
    "Config",
    "load_config",
    "resolve_path",
    "deep_merge",
    "validate_config",
    # checkpointing
    "CheckpointManager",
    "compute_fingerprint",
    "adapter_is_complete",
    "adapter_matches_hash",
    "reusable_adapter",
    # timing / formatting
    "Timer",
    "format_duration",
    "markdown_table",
    # device / memory
    "get_device",
    "get_hf_token",
    "hf_auth_kwargs",
    "free_cuda_memory",
    "cuda_peak_memory_mb",
    "set_global_seed",
    "setup_logging",
    "describe_environment",
    # hashing / sizes / io
    "sha256_any",
    "sha256_bytes",
    "sha256_directory",
    "sha256_file",
    "sha256_path",
    "bytes_to_mb",
    "dir_size_bytes",
    "path_size_bytes",
    "write_json",
]
