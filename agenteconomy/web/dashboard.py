from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from flask import Flask, jsonify, render_template, request
except ImportError:  # pragma: no cover - used by dependency-free remote deploys.
    Flask = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]
    render_template = None  # type: ignore[assignment]
    request = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"
MAX_EVENTS = 1000


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _iso_from_mtime(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        return None


def _record_sort_key(path: Path) -> Tuple[int, int, str]:
    name = path.stem
    prefix, _, raw_num = name.partition("_")
    phase = 0 if prefix == "preheat" else 1
    return (phase, _safe_int(raw_num, 0), path.name)


def _record_files(run_dir: Path) -> List[Path]:
    files = list(run_dir.glob("preheat_*.json")) + list(run_dir.glob("month_*.json"))
    return sorted(files, key=_record_sort_key)


def _latest_mtime(paths: Iterable[Path]) -> float:
    latest = 0.0
    for path in paths:
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


def _nested(mapping: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def summarize_record_file(path: Path) -> Dict[str, Any]:
    payload = _read_json(path)
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    macro = payload.get("macro") if isinstance(payload.get("macro"), dict) else {}
    labor = payload.get("labor_market") if isinstance(payload.get("labor_market"), dict) else {}
    product = payload.get("product_market") if isinstance(payload.get("product_market"), dict) else {}
    government = payload.get("government") if isinstance(payload.get("government"), dict) else {}
    household = payload.get("household") if isinstance(payload.get("household"), dict) else {}
    household_aggregate = (
        household.get("aggregate") if isinstance(household.get("aggregate"), dict) else {}
    )

    gdp = macro.get("gdp_comprehensive") if isinstance(macro.get("gdp_comprehensive"), dict) else {}
    price_index = macro.get("price_index") if isinstance(macro.get("price_index"), dict) else {}
    production = details.get("production") if isinstance(details.get("production"), dict) else {}
    demand = details.get("demand") if isinstance(details.get("demand"), dict) else {}
    consumption = details.get("consumption") if isinstance(details.get("consumption"), dict) else {}
    service_consumption = (
        details.get("service_consumption")
        if isinstance(details.get("service_consumption"), dict)
        else {}
    )
    procurement = details.get("procurement") if isinstance(details.get("procurement"), dict) else {}
    firm_credit = details.get("firm_credit") if isinstance(details.get("firm_credit"), dict) else {}
    pressure = details.get("firm_profit_pressure") if isinstance(details.get("firm_profit_pressure"), dict) else {}
    pressure_agg = pressure.get("aggregate") if isinstance(pressure.get("aggregate"), dict) else {}
    accounting = (
        details.get("accounting_invariants")
        if isinstance(details.get("accounting_invariants"), dict)
        else {}
    )
    accounting_metrics = (
        accounting.get("metrics") if isinstance(accounting.get("metrics"), dict) else {}
    )

    nominal_gdp = _safe_float(gdp.get("nominal_gdp"), _safe_float(macro.get("nominal_gdp")))
    real_gdp = _safe_float(gdp.get("real_gdp"), _safe_float(macro.get("real_gdp")))

    return {
        "file": path.name,
        "mtime": _iso_from_mtime(path),
        "month": _safe_int(payload.get("month")),
        "econ_month": _safe_int(payload.get("econ_month")),
        "preheat": bool(payload.get("preheat", False)),
        "timestamp": payload.get("timestamp"),
        "population": payload.get("population", {}),
        "macro": {
            "nominal_gdp": nominal_gdp,
            "real_gdp": real_gdp,
            "gdp_growth_rate": _safe_float(macro.get("gdp_growth_rate")),
            "real_gdp_growth_rate": _safe_float(macro.get("real_gdp_growth_rate")),
            "price_index": _safe_float(price_index.get("index_100"), 100.0),
            "inflation_rate": _safe_float(macro.get("inflation_rate")),
            "labor_share": _safe_float(macro.get("labor_share")),
            "consumption_rate": _safe_float(macro.get("consumption_rate")),
            "government_rate": _safe_float(macro.get("government_rate")),
        },
        "labor": {
            "total_labor": _safe_float(labor.get("total_labor")),
            "employed_labor": _safe_float(labor.get("employed_labor")),
            "employment_rate": _safe_float(labor.get("employment_rate")),
            "unemployment_rate": _safe_float(labor.get("unemployment_rate")),
            "total_wage_gross": _safe_float(labor.get("total_wage_gross")),
            "total_wage_net": _safe_float(labor.get("total_wage_net")),
            "average_wage": _safe_float(labor.get("average_wage")),
            "job_fill_rate": _safe_float(labor.get("job_fill_rate")),
        },
        "market": {
            "household_purchase_total": _safe_float(product.get("household_purchase_total")),
            "government_procurement_total": _safe_float(product.get("government_procurement_total")),
            "total_output": _safe_float(product.get("total_output")),
            "demand_value": _safe_float(demand.get("total_value")),
            "demand_qty": _safe_float(demand.get("total_qty")),
            "goods_consumption_value": _safe_float(consumption.get("total_value")),
            "service_consumption_value": _safe_float(
                service_consumption.get("total_service_consumption")
            ),
            "procurement_value": _safe_float(procurement.get("total_value")),
        },
        "production": {
            "output_value": _safe_float(
                production.get("total_output_value"),
                _safe_float(production.get("total_value"), _safe_float(product.get("total_output"))),
            ),
            "production_cost": _safe_float(production.get("total_cost")),
            "cash_binding_count": _safe_int(production.get("cash_binding_count")),
            "planning_firms": _safe_int(production.get("planning_firms")),
        },
        "government": government,
        "household": {"aggregate": household_aggregate},
        "firm_credit": {
            "interest_total": _safe_float(firm_credit.get("interest_total")),
            "repayment_total": _safe_float(firm_credit.get("repayment_total")),
            "defaulted_count": _safe_int(firm_credit.get("defaulted_count")),
            "firm_count": len(firm_credit.get("firms", {}) or {}),
        },
        "profit_pressure": {
            "realized_income": _safe_float(pressure_agg.get("realized_income")),
            "realized_expenses": _safe_float(pressure_agg.get("realized_expenses")),
            "realized_profit": _safe_float(pressure_agg.get("realized_profit")),
            "wage_expense": _safe_float(pressure_agg.get("wage_expense")),
            "sales_gap_to_wages": _safe_float(pressure_agg.get("sales_gap_to_wages")),
            "income_to_wage_ratio": _safe_float(pressure_agg.get("income_to_wage_ratio")),
            "firms_income_below_wages": _safe_int(pressure_agg.get("firms_income_below_wages")),
            "top_sales_gap_to_wages": pressure.get("top_sales_gap_to_wages", []),
        },
        "accounting": {
            "ok": bool(accounting.get("ok", False)),
            "error_count": len(accounting.get("errors", []) or []),
            "warning_count": len(accounting.get("warnings", []) or []),
            "ledger_total_cash": _safe_float(accounting_metrics.get("ledger_total_cash")),
            "negative_firm_cash_count": _safe_int(accounting_metrics.get("negative_firm_cash_count")),
            "negative_inventory_count": _safe_int(accounting_metrics.get("negative_inventory_count")),
            "flow_residual": _safe_float(accounting_metrics.get("total_net_flow_residual")),
        },
    }


def read_stage_events(run_dir: Path, limit: int = MAX_EVENTS) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = run_dir / "stage_events.jsonl"
    if not path.exists():
        return [], {
            "latest_event": None,
            "active_stage": None,
            "active_count": 0,
            "completed_count": 0,
            "error_count": 0,
        }

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], {
            "latest_event": None,
            "active_stage": None,
            "active_count": 0,
            "completed_count": 0,
            "error_count": 0,
        }

    if limit > 0:
        lines = lines[-limit:]

    events: List[Dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)

    active: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    completed_count = 0
    error_count = 0
    for event in events:
        key = (event.get("preheat"), event.get("month"), event.get("stage"))
        if event.get("event") == "start":
            active[key] = event
        elif event.get("event") == "end":
            active.pop(key, None)
            completed_count += 1
        if event.get("status") == "error":
            error_count += 1

    active_stage = None
    if active:
        active_stage = sorted(active.values(), key=lambda item: _safe_int(item.get("seq")))[-1]

    return events, {
        "latest_event": events[-1] if events else None,
        "active_stage": active_stage,
        "active_count": len(active),
        "completed_count": completed_count,
        "error_count": error_count,
    }


def iter_run_dirs(output_root: Path) -> List[Path]:
    if not output_root.exists():
        return []
    runs = []
    for path in output_root.rglob("run_*"):
        if not path.is_dir():
            continue
        if _record_files(path) or (path / "stage_events.jsonl").exists():
            runs.append(path)
    return runs


def summarize_run_dir(run_dir: Path, output_root: Path) -> Dict[str, Any]:
    records = _record_files(run_dir)
    stage_path = run_dir / "stage_events.jsonl"
    _, stage = read_stage_events(run_dir, limit=MAX_EVENTS if stage_path.exists() else 0)
    watch_files = list(records)
    if stage_path.exists():
        watch_files.append(stage_path)
    latest = _latest_mtime(watch_files)
    latest_record = summarize_record_file(records[-1]) if records else None
    rel_id = str(run_dir.resolve().relative_to(output_root.resolve()))
    return {
        "id": rel_id,
        "name": run_dir.name,
        "path": str(run_dir),
        "record_count": len(records),
        "month_count": sum(1 for path in records if path.name.startswith("month_")),
        "preheat_count": sum(1 for path in records if path.name.startswith("preheat_")),
        "has_stage_events": stage_path.exists(),
        "stage": stage,
        "active_stage": stage.get("active_stage"),
        "is_active": bool(stage.get("active_stage")),
        "updated_at": datetime.fromtimestamp(latest).isoformat() if latest > 0 else None,
        "latest_record": latest_record,
    }


def _run_priority(summary: Dict[str, Any]) -> Tuple[int, int, str]:
    return (
        1 if summary.get("is_active") else 0,
        1 if _safe_int(summary.get("record_count")) > 0 else 0,
        str(summary.get("updated_at") or ""),
    )


def safe_run_dir(output_root: Path, run_id: str) -> Optional[Path]:
    if not run_id:
        return None
    root = output_root.resolve()
    candidate = (root / run_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def latest_run(output_root: Path) -> Optional[Path]:
    runs = iter_run_dirs(output_root)
    if not runs:
        return None
    summaries = [(run, summarize_run_dir(run, output_root)) for run in runs]
    return max(summaries, key=lambda item: _run_priority(item[1]))[0]


def create_app(output_root: Optional[Path] = None) -> Flask:
    if Flask is None:
        raise RuntimeError("Flask is not installed; use agenteconomy.web.standalone_dashboard")
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).with_name("templates")),
        static_folder=str(Path(__file__).with_name("static")),
    )
    app.config["OUTPUT_ROOT"] = Path(output_root or DEFAULT_OUTPUT_ROOT).resolve()
    app.config["STARTED_AT"] = time.time()

    @app.get("/")
    def index():
        return render_template("dashboard.html")

    @app.get("/api/health")
    def health():
        root = app.config["OUTPUT_ROOT"]
        return jsonify(
            {
                "ok": True,
                "output_root": str(root),
                "output_exists": root.exists(),
                "uptime_seconds": round(time.time() - app.config["STARTED_AT"], 3),
            }
        )

    @app.get("/api/runs")
    def runs():
        root = app.config["OUTPUT_ROOT"]
        limit = max(1, min(_safe_int(request.args.get("limit"), 50), 200))
        items = [summarize_run_dir(run, root) for run in iter_run_dirs(root)]
        items.sort(key=_run_priority, reverse=True)
        return jsonify({"runs": items[:limit], "count": len(items)})

    @app.get("/api/latest")
    def latest():
        root = app.config["OUTPUT_ROOT"]
        run = latest_run(root)
        if run is None:
            return jsonify({"run": None, "records": [], "events": [], "stage": {}})
        records = [summarize_record_file(path) for path in _record_files(run)]
        events, stage = read_stage_events(run)
        return jsonify(
            {
                "run": summarize_run_dir(run, root),
                "records": records,
                "events": events,
                "stage": stage,
            }
        )

    @app.get("/api/run")
    def run_detail():
        root = app.config["OUTPUT_ROOT"]
        run_id = request.args.get("path", "")
        run = safe_run_dir(root, run_id)
        if run is None:
            return jsonify({"error": "run not found"}), 404
        records = [summarize_record_file(path) for path in _record_files(run)]
        events, stage = read_stage_events(run)
        return jsonify(
            {
                "run": summarize_run_dir(run, root),
                "records": records,
                "events": events,
                "stage": stage,
            }
        )

    return app


def main() -> None:
    if Flask is None:
        from agenteconomy.web.standalone_dashboard import main as standalone_main

        standalone_main()
        return

    parser = argparse.ArgumentParser(description="AgentEconomy web dashboard")
    parser.add_argument("--host", default=os.getenv("AGENTECONOMY_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENTECONOMY_WEB_PORT", "7860")))
    parser.add_argument("--output-root", default=os.getenv("AGENTECONOMY_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(Path(args.output_root))
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
