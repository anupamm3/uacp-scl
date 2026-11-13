"""Run Agent Alpha alone against a synthetic ledger dataset and report precision and recall."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.data.make_ledger import find_anomalies_baseline
from uacp.adapters.base import Adapter
from uacp.adapters.fake_adapter import FakeAdapter
from uacp.adapters.registry import get_adapter
from uacp.agents.alpha_research import AlphaResearchAgent, chunk_ledger


def create_fake_adapter_for_ledger(
    ledger: list[dict[str, Any]],
    chunk_size: int = 20,
) -> FakeAdapter:
    """Create a FakeAdapter scripted with deterministic anomaly outputs matching chunks."""
    chunks = chunk_ledger(ledger, chunk_size=chunk_size)
    script: list[dict[str, Any]] = []

    for batch_name, chunk in chunks:
        detected = find_anomalies_baseline(chunk)
        anomalies_payload = [
            {
                "id": a["id"],
                "batch": a.get("batch", batch_name),
                "delta": float(a.get("delta", 0.0)),
                "type": a.get("type", "anomaly"),
                "evidence": f"Rule-based detection: {a.get('type')} in {a.get('id')}",
            }
            for a in detected
        ]
        script.append(
            {
                "audit_complete": True,
                "anomalies": anomalies_payload,
            }
        )

    return FakeAdapter(script=script)


async def run_evaluation(
    alias: str,
    ledger_path: Path,
    ground_truth_path: Path,
    scl_path: Path,
    entries_per_call: int,
    max_calls: int,
) -> int:
    """Run Agent Alpha evaluation and print forensic metrics."""
    if not ledger_path.is_file():
        print(f"Error: Ledger file not found at {ledger_path}", file=sys.stderr)
        return 1

    if not ground_truth_path.is_file():
        print(f"Error: Ground truth file not found at {ground_truth_path}", file=sys.stderr)
        return 1

    ledger: list[dict[str, Any]] = json.loads(ledger_path.read_text(encoding="utf-8"))
    ground_truth: list[dict[str, Any]] = json.loads(
        ground_truth_path.read_text(encoding="utf-8")
    )

    adapter: Adapter
    if alias == "fake":
        adapter = create_fake_adapter_for_ledger(ledger, chunk_size=entries_per_call)
    else:
        try:
            adapter = get_adapter(alias)
        except Exception as exc:
            print(f"Error instantiating adapter for alias '{alias}': {exc}", file=sys.stderr)
            return 1

    agent = AlphaResearchAgent(
        agent_id="alpha",
        scl_path=scl_path,
        adapter=adapter,
        entries_per_batch=entries_per_call,
        max_llm_calls=max_calls,
    )

    print("=" * 60)
    print("Running Agent Alpha Evaluation")
    print(f"  Alias:            {alias}")
    print(f"  Ledger Entries:   {len(ledger)}")
    print(f"  Entries per Call: {entries_per_call}")
    print(f"  Max Calls:        {max_calls}")
    print(f"  Ground Truth:     {len(ground_truth)} anomalies")
    print("=" * 60)

    report, usage, repairs = await agent.audit_ledger(
        ledger=ledger,
        entries_per_batch=entries_per_call,
        max_llm_calls=max_calls,
    )

    # Calculate metrics
    gt_keys = {(item["batch"], item["id"]) for item in ground_truth}
    det_keys = {(f.batch, f.id) for f in report.anomalies}
    tp_keys = gt_keys.intersection(det_keys)

    precision = len(tp_keys) / len(det_keys) if det_keys else 0.0
    recall = len(tp_keys) / len(gt_keys) if gt_keys else 0.0

    gt_type_keys = {(item["id"], item["type"]) for item in ground_truth}
    det_type_keys = {(f.id, f.type) for f in report.anomalies}
    tp_types = gt_type_keys.intersection(det_type_keys)

    type_precision = len(tp_types) / len(det_type_keys) if det_type_keys else 0.0
    type_recall = len(tp_types) / len(gt_type_keys) if gt_type_keys else 0.0

    print("\nExecution Results:")
    print(f"  Status:           {report.status}")
    print(f"  Audit Complete:   {report.audit_complete}")
    print(f"  Detected Count:   {report.total_anomalies}")
    print(f"  Difficulty:       {report.difficulty}/5")
    print(f"  LLM Calls:        {usage.requests}")
    print(f"  Input Tokens:     {usage.input_tokens}")
    print(f"  Output Tokens:    {usage.output_tokens}")
    print(f"  Total Tokens:     {usage.input_tokens + usage.output_tokens}")
    print(f"  Repairs:          {repairs}")

    print("\nMetrics:")
    print(f"  Precision (ID):       {precision:.2%} ({len(tp_keys)}/{len(det_keys)})")
    print(f"  Recall (ID):          {recall:.2%} ({len(tp_keys)}/{len(gt_keys)})")
    print(f"  Precision (ID+Type):  {type_precision:.2%} ({len(tp_types)}/{len(det_type_keys)})")
    print(f"  Recall (ID+Type):     {type_recall:.2%} ({len(tp_types)}/{len(gt_type_keys)})")

    if report.anomalies:
        print("\nDetected Anomalies:")
        for idx, anom in enumerate(report.anomalies, start=1):
            print(
                f"  {idx}. [{anom.batch}] {anom.id} | {anom.type} | "
                f"delta={anom.delta:.2f} | diff={anom.difficulty}"
            )

    print("=" * 60)
    return 0


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate Agent Alpha against ground truth.")
    parser.add_argument(
        "--alias",
        type=str,
        default="fake",
        help="Model alias from config/models.yaml (default: 'fake', e.g. 'alpha_default')",
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default="examples/data/ledger_with_anomalies.json",
        help="Path to anomalous ledger JSON file",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="examples/data/ground_truth.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--scl-path",
        type=str,
        default="examples/financial_auditor.scl",
        help="Path to SCL specification file",
    )
    parser.add_argument(
        "--entries-per-call",
        type=int,
        default=20,
        help="Max transactions per LLM chunk (default: 20)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=10,
        help="Maximum LLM calls permitted (default: 10)",
    )

    args = parser.parse_args()

    exit_code = asyncio.run(
        run_evaluation(
            alias=args.alias,
            ledger_path=Path(args.ledger),
            ground_truth_path=Path(args.ground_truth),
            scl_path=Path(args.scl_path),
            entries_per_call=args.entries_per_call,
            max_calls=args.max_calls,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
