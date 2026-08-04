// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FedChainAudit
 * @notice Immutable audit registry for federated fine-tuning of large language
 *         models, as described in "FedChain: Auditable Federated Fine-Tuning of
 *         Large Language Models using Blockchain and IPFS".
 *
 * @dev The chain never stores model weights. For every local update a client
 *      submits only two commitments:
 *        - `modelHash` : hex SHA-256 digest of the serialized LoRA adapter
 *        - `ipfsCID`   : content identifier of the adapter pinned to IPFS
 *      Anyone can later fetch the adapter by CID, recompute SHA-256 and compare
 *      it against the on-chain record, which yields tamper-evident provenance
 *      for every contribution in every round without leaking training data.
 *
 *      Gas notes: records are append-only and never mutated, so each call costs
 *      one array push plus the dynamic string payloads. `getAuditLogs()` is a
 *      `view` helper for off-chain indexers - it is unbounded and must not be
 *      called from another contract.
 */
contract FedChainAudit {
    /// @notice One immutable provenance record for a single model update.
    struct AuditRecord {
        uint256 round;      // federated round index (1-based)
        string clientId;    // logical participant id, e.g. "client_1" or "server"
        string modelHash;   // SHA-256 of the adapter, lowercase hex
        string ipfsCID;     // IPFS content identifier ("" when IPFS is disabled)
        uint256 timestamp;  // block timestamp at inclusion
        address submitter;  // account that anchored the record
    }

    /// @dev Append-only ledger of every submitted record.
    AuditRecord[] private _auditLogs;

    /// @dev round => indices into `_auditLogs`, for cheap per-round lookup.
    mapping(uint256 => uint256[]) private _recordsByRound;

    /// @dev modelHash => 1-based index into `_auditLogs` (0 means "unknown").
    mapping(string => uint256) private _indexByHash;

    /// @notice Emitted for every anchored update; `round` is indexed for filters.
    event ModelLog(
        uint256 indexed round,
        string clientId,
        string modelHash,
        string ipfsCID,
        uint256 timestamp
    );

    /**
     * @notice Anchor one model update on-chain.
     * @param round     Federated round index this update belongs to.
     * @param clientId  Logical identifier of the submitting participant.
     * @param modelHash Lowercase hex SHA-256 digest of the adapter artefact.
     * @param ipfsCID   IPFS CID of the adapter, or "" when IPFS is disabled.
     */
    function logUpdate(
        uint256 round,
        string memory clientId,
        string memory modelHash,
        string memory ipfsCID
    ) public {
        require(bytes(clientId).length > 0, "FedChainAudit: empty clientId");
        require(bytes(modelHash).length > 0, "FedChainAudit: empty modelHash");

        AuditRecord memory record = AuditRecord({
            round: round,
            clientId: clientId,
            modelHash: modelHash,
            ipfsCID: ipfsCID,
            timestamp: block.timestamp,
            submitter: msg.sender
        });

        _auditLogs.push(record);
        uint256 index = _auditLogs.length - 1;
        _recordsByRound[round].push(index);

        // Keep the first anchoring of a digest authoritative: a replayed hash
        // must not be able to overwrite the original provenance pointer.
        if (_indexByHash[modelHash] == 0) {
            _indexByHash[modelHash] = index + 1;
        }

        emit ModelLog(round, clientId, modelHash, ipfsCID, block.timestamp);
    }

    /// @notice Return the complete audit trail.
    /// @dev Unbounded read; intended for off-chain `eth_call` only.
    function getAuditLogs() public view returns (AuditRecord[] memory) {
        return _auditLogs;
    }

    /// @notice Total number of anchored records.
    function getLogCount() public view returns (uint256) {
        return _auditLogs.length;
    }

    /// @notice Return a single record by its global index.
    function getLog(uint256 index) public view returns (AuditRecord memory) {
        require(index < _auditLogs.length, "FedChainAudit: index out of range");
        return _auditLogs[index];
    }

    /// @notice Return every record anchored for a given federated round.
    function getLogsByRound(uint256 round) public view returns (AuditRecord[] memory) {
        uint256[] storage indices = _recordsByRound[round];
        AuditRecord[] memory out = new AuditRecord[](indices.length);
        for (uint256 i = 0; i < indices.length; i++) {
            out[i] = _auditLogs[indices[i]];
        }
        return out;
    }

    /// @notice Number of records anchored for a given round.
    function getRoundLogCount(uint256 round) public view returns (uint256) {
        return _recordsByRound[round].length;
    }

    /**
     * @notice Membership check used by verifiers to prove an artefact was
     *         actually registered by the federation.
     * @return found True when the digest has been anchored at least once.
     * @return index Global index of the first record carrying that digest.
     */
    function isHashRegistered(string memory modelHash)
        public
        view
        returns (bool found, uint256 index)
    {
        uint256 stored = _indexByHash[modelHash];
        if (stored == 0) {
            return (false, 0);
        }
        return (true, stored - 1);
    }
}
