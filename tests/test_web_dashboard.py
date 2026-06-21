import json
import os
import tempfile
import unittest
from pathlib import Path

from agenteconomy.web.dashboard import (
    create_app,
    latest_run,
    read_stage_events,
    summarize_record_file,
)


class WebDashboardTest(unittest.TestCase):
    def test_summarizes_records_and_stage_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "monthly_records" / "run_20260616_010203"
            run_dir.mkdir(parents=True)
            record_path = run_dir / "month_0001.json"
            record_path.write_text(
                json.dumps(
                    {
                        "month": 1,
                        "econ_month": 2,
                        "preheat": False,
                        "timestamp": "2026-06-16T01:02:03",
                        "population": {"households": 2, "firms": 3},
                        "macro": {
                            "nominal_gdp": 1000.0,
                            "real_gdp": 950.0,
                            "inflation_rate": 0.02,
                            "price_index": {"index_100": 105.0},
                        },
                        "labor_market": {
                            "employment_rate": 0.75,
                            "total_wage_gross": 200.0,
                        },
                        "product_market": {
                            "household_purchase_total": 180.0,
                            "government_procurement_total": 50.0,
                            "total_output": 230.0,
                        },
                        "details": {
                            "firm_credit": {"defaulted_count": 1},
                            "firm_profit_pressure": {
                                "aggregate": {
                                    "realized_income": 230.0,
                                    "wage_expense": 200.0,
                                    "sales_gap_to_wages": 0.0,
                                }
                            },
                            "accounting_invariants": {
                                "ok": True,
                                "metrics": {
                                    "ledger_total_cash": 10000.0,
                                    "total_net_flow_residual": 0.0,
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "stage_events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "seq": 1,
                                "event": "start",
                                "stage": "消费决策",
                                "month": 1,
                                "econ_month": 2,
                                "preheat": False,
                                "status": "ok",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "seq": 2,
                                "event": "end",
                                "stage": "消费决策",
                                "month": 1,
                                "econ_month": 2,
                                "preheat": False,
                                "elapsed_seconds": 0.5,
                                "status": "ok",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = summarize_record_file(record_path)
            self.assertEqual(summary["macro"]["nominal_gdp"], 1000.0)
            self.assertEqual(summary["accounting"]["ok"], True)

            events, stage = read_stage_events(run_dir)
            self.assertEqual(len(events), 2)
            self.assertIsNone(stage["active_stage"])
            self.assertEqual(stage["completed_count"], 1)

            app = create_app(root)
            client = app.test_client()
            runs = client.get("/api/runs").get_json()
            self.assertEqual(runs["count"], 1)
            detail = client.get(
                "/api/run",
                query_string={"path": "monthly_records/run_20260616_010203"},
            ).get_json()
            self.assertEqual(detail["records"][0]["market"]["total_output"], 230.0)
            self.assertEqual(detail["events"][0]["stage"], "消费决策")

    def test_latest_prefers_replayable_run_over_inactive_stage_only_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_run = root / "monthly_records" / "run_20260616_010203"
            replay_run.mkdir(parents=True)
            (replay_run / "month_0001.json").write_text(
                json.dumps({"month": 1, "econ_month": 1, "preheat": False}),
                encoding="utf-8",
            )

            stage_only_run = root / "stage_only" / "run_20260617_010203"
            stage_only_run.mkdir(parents=True)
            (stage_only_run / "stage_events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"seq": 1, "event": "start", "stage": "消费决策"}),
                        json.dumps(
                            {
                                "seq": 2,
                                "event": "end",
                                "stage": "消费决策",
                                "elapsed_seconds": 0.1,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            os.utime(stage_only_run / "stage_events.jsonl", None)

            self.assertEqual(latest_run(root), replay_run)

            app = create_app(root)
            runs = app.test_client().get("/api/runs").get_json()["runs"]
            self.assertEqual(runs[0]["id"], "monthly_records/run_20260616_010203")

    def test_latest_prefers_active_stage_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_run = root / "monthly_records" / "run_20260616_010203"
            replay_run.mkdir(parents=True)
            (replay_run / "month_0001.json").write_text(
                json.dumps({"month": 1, "econ_month": 1, "preheat": False}),
                encoding="utf-8",
            )

            active_run = root / "active" / "run_20260617_010203"
            active_run.mkdir(parents=True)
            (active_run / "stage_events.jsonl").write_text(
                json.dumps({"seq": 1, "event": "start", "stage": "生产补货"}),
                encoding="utf-8",
            )

            self.assertEqual(latest_run(root), active_run)

            app = create_app(root)
            detail = app.test_client().get("/api/latest").get_json()
            self.assertEqual(detail["run"]["id"], "active/run_20260617_010203")
            self.assertEqual(detail["stage"]["active_stage"]["stage"], "生产补货")


if __name__ == "__main__":
    unittest.main()
