"""
FedChain :: Blockchain audit logger
===================================

Thin, defensive Web3.py wrapper around the ``FedChainAudit`` Solidity contract.

Design goals
------------
1. **Never block an experiment.** If ``web3`` is not installed, the RPC node is
   unreachable, or no contract can be deployed, the logger degrades to a
   deterministic in-process *mock chain*. The run completes and every metric is
   flagged ``mode = "mock"`` so results are never silently mislabelled.
2. **Honest measurements.** Transaction latency is measured with a monotonic
   clock around submit + receipt. Gas is read from the receipt in live mode and
   from an explicit, documented cost model in mock mode.
3. **Real integrity.** ``log_model_update`` accepts raw adapter bytes, a file,
   or a PEFT adapter *directory*, and always anchors a SHA-256 digest computed
   over the actual artefact.

Connection strategy (in order):
    web3 importable? -> RPC reachable? -> contract_address given? attach
                                       -> compiled artifact on disk?  deploy
                                       -> py-solc-x available?        compile+deploy
                                       -> otherwise                   MOCK
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.common import bytes_to_mb, path_size_bytes, sha256_any, write_json

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

try:  # pragma: no cover - exercised implicitly by the mock path
    from web3 import Web3

    WEB3_AVAILABLE = True
except Exception:  # ImportError, or a broken native dependency
    Web3 = None  # type: ignore[assignment]
    WEB3_AVAILABLE = False


# =============================================================================
# Contract interface
# =============================================================================
#: ABI of blockchain/contract.sol. Kept inline so the logger can attach to an
#: already-deployed FedChainAudit without a Solidity toolchain being present.
FEDCHAIN_ABI: List[Dict[str, Any]] = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "round", "type": "uint256"},
            {"indexed": False, "internalType": "string", "name": "clientId", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "modelHash", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "ipfsCID", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "name": "ModelLog",
        "type": "event",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "round", "type": "uint256"},
            {"internalType": "string", "name": "clientId", "type": "string"},
            {"internalType": "string", "name": "modelHash", "type": "string"},
            {"internalType": "string", "name": "ipfsCID", "type": "string"},
        ],
        "name": "logUpdate",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getAuditLogs",
        "outputs": [
            {
                "components": [
                    {"internalType": "uint256", "name": "round", "type": "uint256"},
                    {"internalType": "string", "name": "clientId", "type": "string"},
                    {"internalType": "string", "name": "modelHash", "type": "string"},
                    {"internalType": "string", "name": "ipfsCID", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "address", "name": "submitter", "type": "address"},
                ],
                "internalType": "struct FedChainAudit.AuditRecord[]",
                "name": "",
                "type": "tuple[]",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getLogCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "name": "getLog",
        "outputs": [
            {
                "components": [
                    {"internalType": "uint256", "name": "round", "type": "uint256"},
                    {"internalType": "string", "name": "clientId", "type": "string"},
                    {"internalType": "string", "name": "modelHash", "type": "string"},
                    {"internalType": "string", "name": "ipfsCID", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "address", "name": "submitter", "type": "address"},
                ],
                "internalType": "struct FedChainAudit.AuditRecord",
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "round", "type": "uint256"}],
        "name": "getLogsByRound",
        "outputs": [
            {
                "components": [
                    {"internalType": "uint256", "name": "round", "type": "uint256"},
                    {"internalType": "string", "name": "clientId", "type": "string"},
                    {"internalType": "string", "name": "modelHash", "type": "string"},
                    {"internalType": "string", "name": "ipfsCID", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "address", "name": "submitter", "type": "address"},
                ],
                "internalType": "struct FedChainAudit.AuditRecord[]",
                "name": "",
                "type": "tuple[]",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "round", "type": "uint256"}],
        "name": "getRoundLogCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "modelHash", "type": "string"}],
        "name": "isHashRegistered",
        "outputs": [
            {"internalType": "bool", "name": "found", "type": "bool"},
            {"internalType": "uint256", "name": "index", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

_RECORD_FIELDS = ("round", "clientId", "modelHash", "ipfsCID", "timestamp", "submitter")


# =============================================================================
# Records
# =============================================================================
@dataclass
class AuditReceipt:
    """One anchored update plus the systems metrics of anchoring it."""

    round: int
    client_id: str
    model_hash: str
    ipfs_cid: str
    timestamp: int
    mode: str                       # "live" | "mock"
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    effective_gas_price_wei: Optional[int] = None
    tx_cost_eth: Optional[float] = None
    latency_sec: float = 0.0
    confirmation_latency_sec: float = 0.0
    status: str = "success"
    artifact_bytes: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _MockBlock:
    number: int
    timestamp: int


class MockChain:
    """Deterministic in-process stand-in for an EVM node.

    It is *not* a simulator: it records the same payload the contract would, and
    prices each call with an explicit, documented gas model so that the reported
    "gas cost" column is reproducible and clearly synthetic rather than random.

    Cost model (mirrors the real `logUpdate` shape):
        21_000                      base transaction
      + 16 per non-zero calldata byte (EIP-2028)
      + 20_000 per fresh 32-byte storage word written
      +  2_100 event topic/data overhead
    """

    BASE_TX_GAS = 21_000
    CALLDATA_BYTE_GAS = 16
    STORAGE_WORD_GAS = 20_000
    EVENT_GAS = 2_100
    GAS_PRICE_WEI = 1_000_000_000  # 1 gwei, matching Hardhat's default

    def __init__(self, start_block: int = 1):
        self.records: List[Dict[str, Any]] = []
        self.block_number = int(start_block)

    @staticmethod
    def _words(text: str) -> int:
        """Number of 32-byte storage words a dynamic string occupies."""
        return 1 + (len(text.encode("utf-8")) + 31) // 32

    def estimate_gas(self, round_index: int, client_id: str, model_hash: str, ipfs_cid: str) -> int:
        calldata_bytes = (
            32  # uint256 round
            + 3 * 32  # three dynamic offsets
            + sum(32 + ((len(s.encode("utf-8")) + 31) // 32) * 32 for s in (client_id, model_hash, ipfs_cid))
        )
        storage_words = (
            2  # round + timestamp
            + 1  # packed submitter address
            + self._words(client_id)
            + self._words(model_hash)
            + self._words(ipfs_cid)
            + 2  # array length bump + per-round index push
        )
        return (
            self.BASE_TX_GAS
            + self.CALLDATA_BYTE_GAS * calldata_bytes
            + self.STORAGE_WORD_GAS * storage_words
            + self.EVENT_GAS
        )

    def log_update(self, round_index: int, client_id: str, model_hash: str, ipfs_cid: str) -> Dict[str, Any]:
        self.block_number += 1
        timestamp = int(time.time())
        gas_used = self.estimate_gas(round_index, client_id, model_hash, ipfs_cid)
        # Deterministic pseudo tx hash: keccak is unavailable without web3, and
        # SHA-256 over the payload is sufficient for an identifier.
        import hashlib

        payload = f"{round_index}|{client_id}|{model_hash}|{ipfs_cid}|{self.block_number}"
        tx_hash = "0x" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

        record = {
            "round": int(round_index),
            "clientId": client_id,
            "modelHash": model_hash,
            "ipfsCID": ipfs_cid,
            "timestamp": timestamp,
            "submitter": "0x0000000000000000000000000000000000000000",
            "txHash": tx_hash,
            "blockNumber": self.block_number,
            "gasUsed": gas_used,
            "effectiveGasPrice": self.GAS_PRICE_WEI,
        }
        self.records.append(record)
        return record

    def get_audit_logs(self) -> List[Dict[str, Any]]:
        return [{k: rec[k] for k in _RECORD_FIELDS} for rec in self.records]


# =============================================================================
# Logger
# =============================================================================
class BlockchainLogger:
    """Anchors SHA-256 model commitments on an EVM chain (or a mock of one).

    Parameters
    ----------
    rpc_url:
        JSON-RPC endpoint, e.g. ``http://127.0.0.1:8545`` (Hardhat / Anvil / Geth).
    contract_address:
        Address of an already-deployed ``FedChainAudit``. Empty string triggers
        an auto-deploy attempt.
    private_key:
        Hex key used to sign transactions. Empty string uses the node's first
        unlocked account, which is the normal case for a local dev chain.
    force_mock:
        Skip all network probing and go straight to mock mode.
    """

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        contract_address: str = "",
        private_key: str = "",
        chain_id: Optional[int] = None,
        abi: Optional[List[Dict[str, Any]]] = None,
        contract_artifact: Optional[PathLike] = None,
        contract_source: Optional[PathLike] = None,
        tx_timeout: int = 120,
        connect_timeout: int = 5,
        force_mock: bool = False,
    ) -> None:
        self.rpc_url = rpc_url
        self.contract_address = (contract_address or "").strip()
        self.private_key = (private_key or os.environ.get("FEDCHAIN_PRIVATE_KEY", "")).strip()
        self.chain_id = chain_id
        self.abi = abi or FEDCHAIN_ABI
        self.contract_artifact = Path(contract_artifact) if contract_artifact else None
        self.contract_source = Path(contract_source) if contract_source else None
        self.tx_timeout = int(tx_timeout)
        self.connect_timeout = int(connect_timeout)

        self.w3: Any = None
        self.contract: Any = None
        self.account: Optional[str] = None
        self.mode: str = "mock"
        self.mock_chain = MockChain()
        self.receipts: List[AuditReceipt] = []
        self.connection_error: Optional[str] = None

        if force_mock:
            LOGGER.info("BlockchainLogger: mock mode forced by caller.")
            self._enter_mock_mode("forced")
        else:
            self._connect()

    # -- properties ----------------------------------------------------------
    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"

    # -- connection ----------------------------------------------------------
    def _enter_mock_mode(self, reason: str) -> None:
        self.mode = "mock"
        self.connection_error = reason
        self.w3 = None
        self.contract = None
        LOGGER.warning(
            "BlockchainLogger running in MOCK mode (%s). Audit records stay in-process; "
            "reported gas is a deterministic estimate, not a chain measurement.",
            reason,
        )

    def _connect(self) -> None:
        if not WEB3_AVAILABLE:
            self._enter_mock_mode("web3.py is not installed")
            return

        try:
            provider = Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": self.connect_timeout})
            w3 = Web3(provider)
            connected = w3.is_connected() if hasattr(w3, "is_connected") else w3.isConnected()
        except Exception as exc:
            self._enter_mock_mode(f"could not reach RPC {self.rpc_url}: {exc}")
            return

        if not connected:
            self._enter_mock_mode(f"no JSON-RPC node answering at {self.rpc_url}")
            return

        self.w3 = w3
        try:
            if self.chain_id is None:
                self.chain_id = w3.eth.chain_id
        except Exception:
            self.chain_id = None

        if not self._resolve_account():
            self._enter_mock_mode("RPC reachable but no usable account (set private_key or unlock a node account)")
            return

        if not self._bind_contract():
            self._enter_mock_mode("RPC reachable but no FedChainAudit contract could be attached or deployed")
            return

        self.mode = "live"
        LOGGER.info(
            "BlockchainLogger connected: rpc=%s chain_id=%s account=%s contract=%s",
            self.rpc_url,
            self.chain_id,
            self.account,
            self.contract_address,
        )

    def _resolve_account(self) -> bool:
        """Pick the signing account: explicit key first, then an unlocked one."""
        if self.private_key:
            try:
                from eth_account import Account

                acct = Account.from_key(self.private_key)
                self.account = acct.address
                self.w3.eth.default_account = acct.address
                LOGGER.info("Using explicit signing key for account %s", self.account)
                return True
            except Exception as exc:
                LOGGER.warning("Provided private_key is unusable (%s); trying node accounts.", exc)
                self.private_key = ""

        try:
            accounts = list(self.w3.eth.accounts)
        except Exception as exc:
            LOGGER.warning("Could not enumerate node accounts: %s", exc)
            return False

        if not accounts:
            return False

        self.account = accounts[0]
        self.w3.eth.default_account = self.account
        LOGGER.info("Using unlocked node account %s", self.account)
        return True

    def _bind_contract(self) -> bool:
        """Attach to an existing deployment, or deploy a fresh contract."""
        if self.contract_address:
            try:
                address = Web3.to_checksum_address(self.contract_address)
                self.contract = self.w3.eth.contract(address=address, abi=self.abi)
                code = self.w3.eth.get_code(address)
                if not code or code in (b"", b"0x", "0x"):
                    LOGGER.warning("No bytecode at %s - the address holds no contract.", address)
                    return False
                self.contract_address = address
                LOGGER.info("Attached to FedChainAudit at %s", address)
                return True
            except Exception as exc:
                LOGGER.warning("Could not attach to contract %s: %s", self.contract_address, exc)
                return False

        bytecode = self._load_bytecode()
        if not bytecode:
            LOGGER.warning(
                "No contract_address given and no bytecode available "
                "(expected %s, or install py-solc-x to compile %s).",
                self.contract_artifact,
                self.contract_source,
            )
            return False
        return self._deploy(bytecode)

    def _load_bytecode(self) -> Optional[str]:
        """Load deployment bytecode from a build artifact, else compile on the fly."""
        if self.contract_artifact and self.contract_artifact.exists():
            try:
                with open(self.contract_artifact, "r", encoding="utf-8") as handle:
                    artifact = json.load(handle)
                bytecode = artifact.get("bytecode") or artifact.get("bin")
                if isinstance(bytecode, dict):  # solc standard-json shape
                    bytecode = bytecode.get("object")
                if artifact.get("abi"):
                    self.abi = artifact["abi"]
                if bytecode:
                    LOGGER.info("Loaded contract bytecode from %s", self.contract_artifact)
                    return bytecode
            except Exception as exc:
                LOGGER.warning("Could not read artifact %s: %s", self.contract_artifact, exc)

        if self.contract_source and self.contract_source.exists():
            try:
                import solcx  # type: ignore

                try:
                    solcx.set_solc_version("0.8.20")
                except Exception:
                    LOGGER.info("Installing solc 0.8.20 via py-solc-x ...")
                    solcx.install_solc("0.8.20")
                    solcx.set_solc_version("0.8.20")

                source = self.contract_source.read_text(encoding="utf-8")
                compiled = solcx.compile_source(
                    source, output_values=["abi", "bin"], solc_version="0.8.20"
                )
                for name, iface in compiled.items():
                    if name.endswith(":FedChainAudit"):
                        self.abi = iface["abi"]
                        LOGGER.info("Compiled %s with py-solc-x", self.contract_source)
                        return iface["bin"]
                LOGGER.warning("FedChainAudit not found in the compilation output of %s", self.contract_source)
            except ImportError:
                LOGGER.info("py-solc-x is not installed; skipping on-the-fly compilation.")
            except Exception as exc:
                LOGGER.warning("Solidity compilation failed: %s", exc)

        return None

    def _deploy(self, bytecode: str) -> bool:
        try:
            factory = self.w3.eth.contract(abi=self.abi, bytecode=bytecode)
            start = time.perf_counter()
            tx_hash = self._submit(factory.constructor())
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.tx_timeout)
            elapsed = time.perf_counter() - start
            address = receipt["contractAddress"]
            self.contract = self.w3.eth.contract(address=address, abi=self.abi)
            self.contract_address = address
            LOGGER.info(
                "Deployed FedChainAudit at %s (gas=%s, %.3fs)", address, receipt.get("gasUsed"), elapsed
            )
            return True
        except Exception as exc:
            LOGGER.warning("Contract deployment failed: %s", exc)
            return False

    def _submit(self, func_call: Any) -> Any:
        """Send a contract call, signing locally when a private key is set."""
        if self.private_key:
            nonce = self.w3.eth.get_transaction_count(self.account)
            tx_params: Dict[str, Any] = {"from": self.account, "nonce": nonce}
            if self.chain_id is not None:
                tx_params["chainId"] = self.chain_id
            try:
                tx_params["gas"] = int(func_call.estimate_gas({"from": self.account}) * 1.25)
            except Exception:
                tx_params["gas"] = 1_500_000
            tx = func_call.build_transaction(tx_params)
            signed = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
            # web3 v7 renamed `rawTransaction` -> `raw_transaction`.
            raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
            return self.w3.eth.send_raw_transaction(raw)
        return func_call.transact({"from": self.account})

    # -- public API ----------------------------------------------------------
    def compute_hash(self, adapter_bytes_or_file: Union[bytes, bytearray, PathLike]) -> str:
        """SHA-256 of adapter bytes, an adapter file, or an adapter directory."""
        return sha256_any(adapter_bytes_or_file)

    def log_model_update(
        self,
        round: int,
        client_id: str,
        adapter_bytes_or_file: Union[bytes, bytearray, PathLike],
        ipfs_cid: str = "",
    ) -> Dict[str, Any]:
        """Anchor one model update and return its receipt as a dictionary.

        ``adapter_bytes_or_file`` may be raw bytes, a path to
        ``adapter_model.safetensors``, or the adapter directory itself. The
        digest is computed over the artefact as it exists on disk, so the value
        anchored on-chain is exactly what a third party can re-verify after
        fetching the artefact from IPFS.
        """
        try:
            model_hash = self.compute_hash(adapter_bytes_or_file)
        except Exception as exc:
            LOGGER.error("Could not hash the adapter for %s: %s", client_id, exc)
            receipt = AuditReceipt(
                round=int(round),
                client_id=str(client_id),
                model_hash="",
                ipfs_cid=ipfs_cid or "",
                timestamp=int(time.time()),
                mode=self.mode,
                status="hash_failed",
                error=str(exc),
            )
            self.receipts.append(receipt)
            return receipt.to_dict()

        if isinstance(adapter_bytes_or_file, (bytes, bytearray, memoryview)):
            artifact_bytes = len(adapter_bytes_or_file)
        else:
            artifact_bytes = path_size_bytes(adapter_bytes_or_file)

        ipfs_cid = ipfs_cid or ""
        # NOTE: the public parameter is named `round` to match the paper's API;
        # the helpers below rename it so the builtin `round()` stays reachable.
        if self.is_mock:
            receipt = self._log_mock(round, client_id, model_hash, ipfs_cid, artifact_bytes)
        else:
            receipt = self._log_live(round, client_id, model_hash, ipfs_cid, artifact_bytes)

        self.receipts.append(receipt)
        LOGGER.info(
            "[chain:%s] round=%d client=%s hash=%s... cid=%s gas=%s latency=%.3fs",
            self.mode,
            receipt.round,
            receipt.client_id,
            receipt.model_hash[:16],
            receipt.ipfs_cid or "-",
            receipt.gas_used,
            receipt.latency_sec,
        )
        return receipt.to_dict()

    def _log_mock(
        self, round_index: int, client_id: str, model_hash: str, ipfs_cid: str, artifact_bytes: int
    ) -> AuditReceipt:
        start = time.perf_counter()
        record = self.mock_chain.log_update(int(round_index), str(client_id), model_hash, ipfs_cid)
        latency = time.perf_counter() - start
        gas_used = record["gasUsed"]
        price = record["effectiveGasPrice"]
        return AuditReceipt(
            round=int(round_index),
            client_id=str(client_id),
            model_hash=model_hash,
            ipfs_cid=ipfs_cid,
            timestamp=record["timestamp"],
            mode="mock",
            tx_hash=record["txHash"],
            block_number=record["blockNumber"],
            gas_used=gas_used,
            effective_gas_price_wei=price,
            tx_cost_eth=round_eth(gas_used * price),
            latency_sec=round(latency, 6),
            confirmation_latency_sec=0.0,
            status="success",
            artifact_bytes=artifact_bytes,
        )

    def _log_live(
        self, round_index: int, client_id: str, model_hash: str, ipfs_cid: str, artifact_bytes: int
    ) -> AuditReceipt:
        start = time.perf_counter()
        try:
            call = self.contract.functions.logUpdate(
                int(round_index), str(client_id), model_hash, ipfs_cid
            )
            tx_hash = self._submit(call)
            submit_elapsed = time.perf_counter() - start

            confirm_start = time.perf_counter()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.tx_timeout)
            confirm_elapsed = time.perf_counter() - confirm_start
            total_elapsed = time.perf_counter() - start

            gas_used = int(tx_receipt.get("gasUsed", 0))
            gas_price = int(tx_receipt.get("effectiveGasPrice", 0) or 0)
            if not gas_price:
                try:
                    gas_price = int(self.w3.eth.get_transaction(tx_hash).get("gasPrice", 0) or 0)
                except Exception:
                    gas_price = 0

            block_number = int(tx_receipt.get("blockNumber", 0))
            try:
                block_timestamp = int(self.w3.eth.get_block(block_number)["timestamp"])
            except Exception:
                block_timestamp = int(time.time())

            status = "success" if int(tx_receipt.get("status", 1)) == 1 else "reverted"
            if status != "success":
                LOGGER.error("Transaction for %s reverted on-chain.", client_id)

            return AuditReceipt(
                round=int(round_index),
                client_id=str(client_id),
                model_hash=model_hash,
                ipfs_cid=ipfs_cid,
                timestamp=block_timestamp,
                mode="live",
                tx_hash=tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash),
                block_number=block_number,
                gas_used=gas_used,
                effective_gas_price_wei=gas_price,
                tx_cost_eth=round_eth(gas_used * gas_price),
                latency_sec=round(total_elapsed, 6),
                confirmation_latency_sec=round(confirm_elapsed, 6),
                status=status,
                artifact_bytes=artifact_bytes,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            LOGGER.error("On-chain logging failed for %s (%s); recording the failure.", client_id, exc)
            return AuditReceipt(
                round=int(round_index),
                client_id=str(client_id),
                model_hash=model_hash,
                ipfs_cid=ipfs_cid,
                timestamp=int(time.time()),
                mode="live",
                latency_sec=round(elapsed, 6),
                status="failed",
                artifact_bytes=artifact_bytes,
                error=str(exc),
            )

    def get_audit_logs(self) -> List[Dict[str, Any]]:
        """Read the full audit trail back from the chain (or the mock ledger)."""
        if self.is_mock:
            return self.mock_chain.get_audit_logs()
        try:
            raw = self.contract.functions.getAuditLogs().call()
        except Exception as exc:
            LOGGER.warning("getAuditLogs() failed: %s", exc)
            return []
        return [dict(zip(_RECORD_FIELDS, tuple(entry))) for entry in raw]

    def verify_artifact(
        self, adapter_bytes_or_file: Union[bytes, bytearray, PathLike], expected_hash: str
    ) -> bool:
        """Re-hash an artefact and compare it with an anchored commitment."""
        try:
            actual = self.compute_hash(adapter_bytes_or_file)
        except Exception as exc:
            LOGGER.error("Verification failed - could not hash artefact: %s", exc)
            return False
        ok = actual.lower() == str(expected_hash).lower()
        if not ok:
            LOGGER.error("INTEGRITY FAILURE: expected %s, computed %s", expected_hash, actual)
        return ok

    def restore_receipts(self, receipts: List[Dict[str, Any]]) -> int:
        """Re-seed receipts from a checkpoint after a crash.

        On resume the transactions anchored before the crash are real and must
        still count towards total gas and latency, but they must not be
        re-submitted. Rehydrating them here keeps ``get_metrics_summary()``
        reporting the whole run rather than only the current session.
        """
        restored = 0
        known_fields = {f for f in AuditReceipt.__dataclass_fields__}  # type: ignore[attr-defined]
        for entry in receipts or []:
            try:
                self.receipts.append(AuditReceipt(**{k: v for k, v in entry.items() if k in known_fields}))
                restored += 1
            except Exception as exc:
                LOGGER.warning("Skipping unreadable checkpointed receipt: %s", exc)
        if restored:
            LOGGER.info("Restored %d blockchain receipt(s) from the checkpoint.", restored)
        return restored

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Aggregate systems metrics across every transaction of the run."""
        successful = [r for r in self.receipts if r.status == "success"]
        latencies = [r.latency_sec for r in successful]
        gas_values = [r.gas_used for r in successful if r.gas_used is not None]
        costs = [r.tx_cost_eth for r in successful if r.tx_cost_eth is not None]

        return {
            "mode": self.mode,
            "rpc_url": self.rpc_url,
            "chain_id": self.chain_id,
            "contract_address": self.contract_address or None,
            "account": self.account,
            "connection_note": self.connection_error,
            "num_transactions": len(self.receipts),
            "num_successful": len(successful),
            "num_failed": len(self.receipts) - len(successful),
            "total_latency_sec": round(sum(latencies), 6),
            "avg_latency_sec": round(sum(latencies) / len(latencies), 6) if latencies else 0.0,
            "max_latency_sec": round(max(latencies), 6) if latencies else 0.0,
            "min_latency_sec": round(min(latencies), 6) if latencies else 0.0,
            "total_gas_used": int(sum(gas_values)) if gas_values else 0,
            "avg_gas_used": int(sum(gas_values) / len(gas_values)) if gas_values else 0,
            "total_cost_eth": round(sum(costs), 12) if costs else 0.0,
            "total_anchored_mb": bytes_to_mb(sum(r.artifact_bytes for r in self.receipts)),
        }

    def export_audit_trail(self, path: PathLike) -> Path:
        """Write receipts + the on-chain view of the trail to a JSON file."""
        payload = {
            "summary": self.get_metrics_summary(),
            "receipts": [r.to_dict() for r in self.receipts],
            "chain_state": self.get_audit_logs(),
        }
        out = write_json(path, payload)
        LOGGER.info("Audit trail exported to %s", out)
        return out

    def close(self) -> None:
        """Release the provider handle (no-op in mock mode)."""
        self.w3 = None
        self.contract = None


def round_eth(wei: int) -> float:
    """Convert wei to ether with enough precision for cheap dev-chain txs."""
    return round(float(wei) / 1e18, 12)
