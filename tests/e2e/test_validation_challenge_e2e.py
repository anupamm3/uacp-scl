"""End-to-end integration and replay tests for the validation challenge pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from uacp.adapters.fake_adapter import FakeAdapter
from uacp.cli import uacp_app
from uacp.runner.validation_challenge import run_validation_challenge

runner = CliRunner()


@pytest.mark.asyncio
async def test_validation_challenge_scripted_e2e(tmp_path: Path) -> None:
    """Run full forensic pipeline with scripted FakeAdapters in CI."""
    alpha_script: list[str | dict[str, Any] | Exception] = [
        # Chunk 1: Detects the 4 known anomalies
        {
            "audit_complete": True,
            "anomalies": [
                {
                    "id": "TX-B01-0002",
                    "batch": "BATCH-2026-Q1",
                    "type": "rounding_drift",
                    "delta": 0.03,
                    "evidence": "Rounding drift against counterpart",
                },
                {
                    "id": "TX-B01-0003",
                    "batch": "BATCH-2026-Q1",
                    "type": "duplicate_id",
                    "delta": 0.0,
                    "evidence": "Duplicate transaction identifier",
                },
                {
                    "id": "TX-B01-0005",
                    "batch": "BATCH-2026-Q1",
                    "type": "sign_flip",
                    "delta": 11060.24,
                    "evidence": "Inverted debit/credit sign",
                },
                {
                    "id": "TX-B01-0007",
                    "batch": "BATCH-2026-Q1",
                    "type": "missing_counterpart",
                    "delta": 13388.08,
                    "evidence": "Missing balancing counterpart",
                },
            ],
        },
        # Chunk 2
        {"audit_complete": True, "anomalies": []},
        # Chunk 3
        {"audit_complete": True, "anomalies": []},
    ]

    remediation_code = """def mitigate(ledger: list[dict]) -> tuple[list[dict], list[dict]]:
    import copy
    seen_ids = set()
    deduped = []
    changes = []
    for tx in ledger:
        if tx['id'] in seen_ids:
            changes.append({'action': 'remove_duplicate', 'id': tx['id']})
            continue
        seen_ids.add(tx['id'])
        deduped.append(copy.deepcopy(tx))
    lookup = {tx['id']: tx for tx in deduped}
    for tx in deduped:
        if tx['id'] == 'TX-B01-0002':
            tx['credit'] = 9609.43
            changes.append({'action': 'fix_drift', 'id': tx['id']})
        elif tx['id'] == 'TX-B01-0005':
            tx['debit'] = 11060.24
            tx['credit'] = 0.0
            changes.append({'action': 'fix_sign_flip', 'id': tx['id']})
    if 'TX-B01-0008' not in lookup:
        new_entry = {
            'id': 'TX-B01-0008',
            'batch': 'BATCH-2026-Q1',
            'date': '2026-01-05',
            'account': '2000-AccountsPayable',
            'debit': 0.0,
            'credit': 13388.08,
            'counterpart_id': 'TX-B01-0007'
        }
        deduped.append(new_entry)
        changes.append({'action': 'add_counterpart', 'id': 'TX-B01-0008'})
    return deduped, changes
"""

    beta_script: list[str | dict[str, Any] | Exception] = [
        {
            "python_code": remediation_code,
            "rationale": "Deduplicate, fix rounding drift, and synthesize counterpart.",
            "risks": ["Post-remediation reconciliation required for synthetic counterpart."],
        }
    ]

    alpha_adapter = FakeAdapter(script=alpha_script, name="fake_alpha")
    beta_adapter = FakeAdapter(script=beta_script, name="fake_beta")

    metrics_file = tmp_path / "metrics.json"

    exit_code, metrics, final_report = await run_validation_challenge(
        alpha_adapter_override=alpha_adapter,
        beta_adapter_override=beta_adapter,
        metrics_out=metrics_file,
    )

    # 1. Exit code must be 0 (schema validation passed)
    assert exit_code == 0
    assert metrics is not None
    assert final_report is not None

    # 2. Precision & Recall against ground truth
    assert metrics.alpha_precision == 1.0
    assert metrics.alpha_recall == 1.0

    # 3. Anomalies mitigated
    assert metrics.anomalies_mitigated_fraction == 1.0
    assert metrics.initial_anomalies_count == 4
    assert metrics.remaining_anomalies_count == 0

    # 4. Schema and Whiteboard state
    assert metrics.passed_schema is True
    assert metrics.final_report_status == "success"
    assert metrics_file.is_file()


@pytest.mark.asyncio
async def test_validation_challenge_record_and_replay_parity(tmp_path: Path) -> None:
    """Prove that replaying a recorded run reproduces the exact same metrics and output."""
    record_file = tmp_path / "recorded_challenge.json"

    # Step 1: Run in offline mode with recording enabled
    exit1, metrics1, report1 = await run_validation_challenge(
        offline=True,
        record_file=record_file,
    )
    assert exit1 == 0
    assert metrics1 is not None
    assert report1 is not None
    assert record_file.is_file()

    # Step 2: Run in replay mode reading the recorded file
    exit2, metrics2, report2 = await run_validation_challenge(
        replay_file=record_file,
    )
    assert exit2 == 0
    assert metrics2 is not None
    assert report2 is not None

    # Step 3: Exact parity assertions
    assert metrics1.alpha_precision == metrics2.alpha_precision
    assert metrics1.alpha_recall == metrics2.alpha_recall
    assert metrics1.anomalies_mitigated_fraction == metrics2.anomalies_mitigated_fraction
    assert metrics1.total_tokens.total_tokens == metrics2.total_tokens.total_tokens
    assert metrics1.total_hops == metrics2.total_hops
    assert metrics1.total_repairs == metrics2.total_repairs
    assert metrics1.passed_schema == metrics2.passed_schema
    assert metrics1.final_report_status == metrics2.final_report_status
    assert report1["code_sha256"] == report2["code_sha256"]
    assert report1["anomalies_found"] == report2["anomalies_found"]
    assert report1["anomalies_mitigated"] == report2["anomalies_mitigated"]


def test_cli_validation_challenge_offline() -> None:
    """Test `uacp run validation-challenge --offline` command exits cleanly."""
    result = runner.invoke(uacp_app, ["run", "validation-challenge", "--offline"])
    assert result.exit_code == 0
