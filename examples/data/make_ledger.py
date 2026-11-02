"""Deterministic synthetic ledger generator and baseline anomaly detector."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

DEBIT_ACCOUNTS = [
    "1010-Cash",
    "1200-AccountsReceivable",
    "1500-Equipment",
    "5100-OfficeSupplies",
    "5200-ConsultingFees",
    "5300-TravelExpense",
    "5400-SoftwareSubscriptions",
]

CREDIT_ACCOUNTS = [
    "2000-AccountsPayable",
    "2100-AccruedLiabilities",
    "3000-CommonStock",
    "4000-ServiceRevenue",
    "4100-ProductSales",
    "4200-InterestIncome",
]


def generate_clean_ledger(
    batches: int = 3,
    entries_per_batch: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a clean, balanced synthetic ledger of double-entry transactions.

    Each economic transaction consists of exactly two balanced entries:
    one debit and one credit linking to each other via counterpart_id.

    Args:
        batches: Number of transaction batches.
        entries_per_batch: Total entries per batch (must be an even number).
        seed: Random seed for deterministic generation.

    Returns:
        List of clean ledger transaction dictionaries.
    """
    if entries_per_batch % 2 != 0:
        raise ValueError(f"entries_per_batch must be even, got {entries_per_batch}")

    rng = random.Random(seed)
    ledger: list[dict[str, Any]] = []
    pairs_per_batch = entries_per_batch // 2

    for b in range(1, batches + 1):
        batch_name = f"BATCH-2026-Q{b}"
        for p in range(1, pairs_per_batch + 1):
            day = (p % 28) + 1
            date_str = f"2026-{b:02d}-{day:02d}"
            amount = round(rng.uniform(50.0, 15000.0), 2)

            debit_id = f"TX-B{b:02d}-{2 * p - 1:04d}"
            credit_id = f"TX-B{b:02d}-{2 * p:04d}"

            debit_acc = rng.choice(DEBIT_ACCOUNTS)
            credit_acc = rng.choice(CREDIT_ACCOUNTS)

            # Entry 1: Debit
            ledger.append(
                {
                    "id": debit_id,
                    "batch": batch_name,
                    "date": date_str,
                    "account": debit_acc,
                    "debit": amount,
                    "credit": 0.0,
                    "counterpart_id": credit_id,
                }
            )

            # Entry 2: Credit (counterpart)
            ledger.append(
                {
                    "id": credit_id,
                    "batch": batch_name,
                    "date": date_str,
                    "account": credit_acc,
                    "debit": 0.0,
                    "credit": amount,
                    "counterpart_id": debit_id,
                }
            )

    return ledger


def seed_anomalies(
    clean_ledger: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Seed known anomalies into a clean ledger and generate ground truth records.

    Anomalies seeded:
    1. rounding_drift: small decimal discrepancy between counterpart entries.
    2. duplicate_id: duplicate transaction identifier reused.
    3. sign_flip: debit/credit columns inverted or invalid polarity.
    4. missing_counterpart: entry pointing to non-existent counterpart id.

    Args:
        clean_ledger: Clean double-entry transactions.
        seed: Random seed for deterministic selection.

    Returns:
        Tuple of (ledger_with_anomalies, ground_truth_list).
    """
    ledger = copy.deepcopy(clean_ledger)
    ground_truth: list[dict[str, Any]] = []

    if len(ledger) < 8:
        raise ValueError("Ledger too small to seed 4 distinct anomalies (minimum 8 entries).")

    # 1. Rounding drift on entry at index 1 (Pair 1, Credit entry)
    drift_target = ledger[1]
    drift_delta = 0.03
    drift_target["credit"] = round(drift_target["credit"] + drift_delta, 2)
    ground_truth.append(
        {
            "id": drift_target["id"],
            "batch": drift_target["batch"],
            "type": "rounding_drift",
            "delta": drift_delta,
        }
    )

    # 2. Duplicate ID: duplicate entry at index 2 (Pair 2, Debit entry)
    dup_source = ledger[2]
    dup_clone = copy.deepcopy(dup_source)
    dup_clone["account"] = "5100-OfficeSupplies"
    ledger.append(dup_clone)
    ground_truth.append(
        {
            "id": dup_source["id"],
            "batch": dup_source["batch"],
            "type": "duplicate_id",
            "delta": 0.0,
        }
    )

    # 3. Sign flip on entry at index 4 (Pair 3, Debit entry)
    # Convert its debit into credit so both Pair 3 entries are credits
    flip_target = ledger[4]
    flip_amt = flip_target["debit"]
    flip_target["credit"] = flip_amt
    flip_target["debit"] = 0.0
    ground_truth.append(
        {
            "id": flip_target["id"],
            "batch": flip_target["batch"],
            "type": "sign_flip",
            "delta": flip_amt,
        }
    )

    # 4. Missing counterpart on entry at index 6 (Pair 4, Debit entry)
    # Remove entry at index 7 (its credit counterpart) from ledger
    missing_target = ledger[6]
    counterpart_to_remove = ledger[7]["id"]
    ledger = [e for e in ledger if e["id"] != counterpart_to_remove]
    ground_truth.append(
        {
            "id": missing_target["id"],
            "batch": missing_target["batch"],
            "type": "missing_counterpart",
            "delta": missing_target["debit"],
        }
    )

    return ledger, ground_truth


def find_anomalies_baseline(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic plain-Python rule checker for ledger anomalies.

    Checks:
    - Duplicate transaction IDs.
    - Missing counterpart entries.
    - Sign flips / polarity mismatches (both counterpart entries on same side).
    - Rounding drift (small decimal differences between counterpart entries).

    Args:
        ledger: List of transaction dictionaries.

    Returns:
        List of detected anomalies (id, batch, type, delta).
    """
    anomalies: list[dict[str, Any]] = []

    # 1. Duplicate ID check
    id_counts: dict[str, int] = {}
    for entry in ledger:
        id_counts[entry["id"]] = id_counts.get(entry["id"], 0) + 1

    duplicate_ids = {tx_id for tx_id, count in id_counts.items() if count > 1}
    for entry in ledger:
        if entry["id"] in duplicate_ids and not any(
            a["id"] == entry["id"] and a["type"] == "duplicate_id" for a in anomalies
        ):
            anomalies.append(
                {
                    "id": entry["id"],
                    "batch": entry["batch"],
                    "type": "duplicate_id",
                    "delta": 0.0,
                }
            )

    # Lookup mapping id -> entry
    all_tx_ids = set(id_counts.keys())
    lookup: dict[str, dict[str, Any]] = {}
    for entry in ledger:
        if entry["id"] not in lookup:
            lookup[entry["id"]] = entry

    # 2. Check counterpart pairing, polarity, and rounding
    for entry in ledger:
        if entry["id"] in duplicate_ids:
            continue

        c_id = entry.get("counterpart_id", "")

        # Missing counterpart check
        if c_id not in all_tx_ids:
            amt = entry["debit"] if entry["debit"] > 0 else entry["credit"]
            anomalies.append(
                {
                    "id": entry["id"],
                    "batch": entry["batch"],
                    "type": "missing_counterpart",
                    "delta": amt,
                }
            )
            continue

        counterpart = lookup[c_id]

        # Sign flip / polarity check:
        # Standard double-entry: one is debit (>0, credit=0), counterpart is credit (>0, debit=0)
        is_debit = entry["debit"] > 0 and entry["credit"] == 0
        is_credit = entry["credit"] > 0 and entry["debit"] == 0

        c_is_debit = counterpart["debit"] > 0 and counterpart["credit"] == 0
        c_is_credit = counterpart["credit"] > 0 and counterpart["debit"] == 0

        if (is_credit and c_is_credit) or (is_debit and c_is_debit):
            # Sign flip: report the modified entry (or one per pair)
            if not any(
                a["type"] == "sign_flip" and a["id"] in (entry["id"], c_id) for a in anomalies
            ):
                amt = entry["credit"] if is_credit else entry["debit"]
                anomalies.append(
                    {
                        "id": entry["id"],
                        "batch": entry["batch"],
                        "type": "sign_flip",
                        "delta": amt,
                    }
                )
            continue

        # Rounding drift check
        if is_debit and c_is_credit:
            drift = round(entry["debit"] - counterpart["credit"], 4)
            if abs(drift) > 1e-4:
                anomalies.append(
                    {
                        "id": counterpart["id"],
                        "batch": counterpart["batch"],
                        "type": "rounding_drift",
                        "delta": round(counterpart["credit"] - entry["debit"], 2),
                    }
                )

    return anomalies


def main() -> None:
    """CLI entrypoint to generate ledger datasets."""
    parser = argparse.ArgumentParser(description="Generate synthetic financial ledger datasets.")
    parser.add_argument("--batches", type=int, default=3, help="Number of batches (default: 3)")
    parser.add_argument(
        "--entries-per-batch",
        type=int,
        default=20,
        help="Entries per batch (default: 20, must be even)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directory to write output JSON files",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_ledger = generate_clean_ledger(
        batches=args.batches,
        entries_per_batch=args.entries_per_batch,
        seed=args.seed,
    )
    anomalous_ledger, ground_truth = seed_anomalies(clean_ledger, seed=args.seed)

    clean_path = out_dir / "ledger_clean.json"
    anomalous_path = out_dir / "ledger_with_anomalies.json"
    gt_path = out_dir / "ground_truth.json"

    clean_path.write_text(json.dumps(clean_ledger, indent=2), encoding="utf-8")
    anomalous_path.write_text(json.dumps(anomalous_ledger, indent=2), encoding="utf-8")
    gt_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    print(f"Generated {len(clean_ledger)} clean transactions -> {clean_path}")
    print(f"Generated {len(anomalous_ledger)} anomalous transactions -> {anomalous_path}")
    print(f"Generated {len(ground_truth)} ground truth anomalies -> {gt_path}")


if __name__ == "__main__":
    main()
