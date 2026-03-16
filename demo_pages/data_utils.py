"""共享数据加载工具"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "output" / "monthly_records" / "run_20260227_065759"


@st.cache_data
def load_monthly_data() -> List[Dict[str, Any]]:
    files = sorted(DATA_DIR.glob("month_*.json"))
    data = []
    for f in files:
        with open(f, "r") as fp:
            data.append(json.load(fp))
    data.sort(key=lambda x: x.get("econ_month", 0))
    return data


@st.cache_data
def load_ablation_data() -> List[Dict[str, Any]]:
    p = Path(__file__).resolve().parents[1] / "output" / "consumption_test_results.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
