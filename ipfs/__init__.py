"""Decentralized storage layer for FedChain (Pinata / Kubo / mock IPFS)."""

from ipfs.pinata_client import (
    IPFSManager,
    TransferRecord,
    base58_encode,
    compute_cidv0,
)

__all__ = ["IPFSManager", "TransferRecord", "compute_cidv0", "base58_encode"]
