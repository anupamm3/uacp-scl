"""Unit tests for synthetic ledger generator and baseline anomaly detector."""

from __future__ import annotations

import json
from pathlib import Path

from examples.data.make_ledger import (
    find_anomalies_baseline,
    generate_clean_ledger,
    seed_anomalies,
)


def test_clean_ledger_generation_determinism() -> None:
    """Verify clean ledger generation produces identical output for the same seed."""
    ledger1 = generate_clean_ledger(batches=3, entries_per_batch=20, seed=42)
    ledger2 = generate_clean_ledger(batches=3, entries_per_batch=20, seed=42)
    assert ledger1 == ledger2
    assert len(ledger1) == 60


def test_seed_anomalies_determinism() -> None:
    """Verify anomaly seeding is deterministic given the same seed and clean ledger."""
    clean1 = generate_clean_ledger(batches=3, entries_per_batch=20, seed=42)
    clean2 = generate_clean_ledger(batches=3, entries_per_batch=20, seed=42)

    anom1, gt1 = seed_anomalies(clean1, seed=42)
    anom2, gt2 = seed_anomalies(clean2, seed=42)

    assert anom1 == anom2
    assert gt1 == gt2
    assert len(gt1) == 4


def test_clean_ledger_has_zero_anomalies() -> None:
    """Verify baseline detector reports zero anomalies on a clean double-entry ledger."""
    clean = generate_clean_ledger(batches=3, entries_per_batch=20, seed=42)
    anomalies = find_anomalies_baseline(clean)
    assert anomalies == []


def test_baseline_recovers_all_seeded_anomalies() -> None:
    """Verify baseline detector achieves 100% recall against seeded ground truth."""
    clean = generate_clean_ledger(batches=3, entries_per_batch=20, seed=42)
    anomalous, ground_truth = seed_anomalies(clean, seed=42)

    detected = find_anomalies_baseline(anomalous)

    gt_pairs = {(item["id"], item["type"]) for item in ground_truth}
    detected_pairs = {(item["id"], item["type"]) for item in detected}

    # Verify 100% recall
    assert gt_pairs.issubset(detected_pairs)
    # Verify exact match (no spurious detections)
    assert gt_pairs == detected_pairs


def test_generated_json_files_exist_and_validate() -> None:
    """Verify pre-generated JSON files in examples/data/ match ground truth."""
    data_dir = Path("examples/data")
    clean_path = data_dir / "ledger_clean.json"
    anom_path = data_dir / "ledger_with_anomalies.json"
    gt_path = data_dir / "ground_truth.json"

    assert clean_path.is_file()
    assert anom_path.is_file()
    assert gt_path.is_file()

    clean_data = json.loads(clean_path.read_text(encoding="utf-8"))
    anom_data = json.loads(anom_path.read_text(encoding="utf-8"))
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))

    assert len(clean_data) == 60
    assert len(gt_data) == 4
    assert find_anomalies_baseline(clean_data) == []

    detected = find_anomalies_baseline(anom_data)
    gt_pairs = {(x["id"], x["type"]) for x in gt_data}
    detected_pairs = {(x["id"], x["type"]) for x in detected}
    assert gt_pairs == detected_pairs

