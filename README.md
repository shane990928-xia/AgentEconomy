# AgentEconomy

A unified LLM-based economic simulation platform with three independent research modules.

---

## Installation

```bash
# Install dependencies
poetry install

# Activate environment
source .venv/bin/activate
```

### Additional dependency for Supply Chain module
```bash
pip install pyarrow
```

---

## Quick Start

All modules are launched through the unified entry point `main.py` via the `--module` flag:

```bash
python main.py --module <agenteconomy|marketsim|supply_chain> [module-specific args]
```

---

## Module 1: AgentEconomy — Macroeconomic Simulation

LLM-driven agent-based macroeconomic simulation with 300 heterogeneous households and 66 firms.

### Run

```bash
# Normal run
python main.py --config config/config_normal.yaml

# Resume from latest checkpoint
python main.py --resume-latest

# Resume from specific checkpoint
python main.py --resume path/to/checkpoint.pkl.gz
```

### Configuration

Edit `config/config_normal.yaml` to control:
- `num_months` / `preheat_months` — simulation duration
- `num_households` / `num_firms` — agent population
- Tax brackets, VAT, corporate tax, interest rate
- LLM concurrency (`max_llm_concurrent`)

### LLM Configuration (`.env`)

```bash
OPENAI_API_KEY=sk-...
BASE_URL=http://your-api-endpoint/v1/
SIMPLE_MODEL=openai/gpt-4.1-nano      # Step0/1/3 consumption decisions
STRONG_MODEL=openai/gpt-5 # (reserved for future use)
```

---

## Module 2: MarketSim — Stock Market Simulation

Large-scale stock market simulation with LLM-powered institutional investors in a NASDAQ-like continuous double auction environment (15k+ agents).

### Run

```bash
python main.py --module marketsim \
    -c rsmtry_LLM3 \
    -t JNJ \
    -d 20250402 \
    -s 1234 \
    -l rmsctry_LLM3 \
    --enable-llm-cache
```

### Key Arguments

| Argument | Description | Example |
|---|---|---|
| `-c, --config` | Config module name | `rsmtry_LLM3` |
| `-t, --ticker` | Stock symbol | `JNJ` |
| `-d, --historical-date` | Trading date (YYYYMMDD) | `20250402` |
| `-s, --seed` | Random seed | `1234` |
| `-l, --log_dir` | Log directory | `rmsctry_LLM3` |
| `--enable-llm-cache` | Cache LLM responses | flag |
| `--start-time` | Market open time | `09:30:00` |
| `--end-time` | Market close time | `16:00:00` |

### LLM Configuration (`.env`)

```bash
DEEPSEEK_API_KEY=sk-...           # shared with OPENAI_API_KEY
BASE_URL=http://your-api-endpoint/v1/
# MARKETSIM_THINK_MODEL=deepseek-reasoner    # optional override
# MARKETSIM_GENERATE_MODEL=deepseek-chat     # optional override
```

### Agent Types
- **LLM Agents** (ManagerAgent): Institutional investors powered by LLM
- **Noise Agents** (×12,000): Retail traders with random behavior
- **Value Agents** (×100): Fundamental-value-based traders
- **Trade Agents** (×2,950): Rule-based traders
- **Market Maker Agents** (×4): Adaptive POV market makers
- **Momentum Agents** (×50): Trend-following traders

### Available Config Modules (`marketsim/config/`)

| Config | Description |
|---|---|
| `rsmtry_LLM3` | LLM agents + full agent mix |
| `rsmtry_LLM4` | Extended LLM config |
| `rsmtry_LLM5` | Multi-day variant |

---

## Module 3: Supply Chain — Supply Chain Research

Three independent supply-chain research experiments using LLM-based company profiling and supplier evaluation.

### Run

```bash
# Experiment 1: Static Network Reconstruction
python main.py --module supply_chain --experiment network \
    --years 2018,2019,2020 \
    --methods llm ml random \
    --result_dir supply_chain/result/network

# Experiment 2: Evolutionary Network Simulation (multi-year)
python main.py --module supply_chain --experiment network_continuous \
    --sandbox_folder supply_chain/data \
    --start_year 2016 --end_year 2020 \
    --methods llm ml random

# Experiment 3: Single-entity Supplier Selection
python main.py --module supply_chain --experiment supplier \
    --entity_id "000C7F-E" \
    --year 2020 \
    --use_llm

# Debug mode (no real LLM calls)
python main.py --module supply_chain --experiment supplier \
    --entity_id "000C7F-E" --year 2020 --debug
```

### Experiment Overview

#### `network` — Static Network Reconstruction
Reconstructs supply-chain networks for given years by comparing LLM, ML, and random baseline methods.

| Stage | Description |
|---|---|
| Stage 1 | Build hub-node candidate pools |
| Stage 2 | Generate company data profiles |
| Stage 3a | LLM procurement personality generation |
| Stage 3b | ML training data extraction |
| Stage 4 | LLM supplier assessment |
| Stage 5a/b/c | LLM / ML / Random supplier selection |
| Stage 6 | Network reconstruction & analysis |

Key arguments: `--years`, `--methods {llm,ml,random}`, `--debug`, `--skip_existing`, `--max_workers`

#### `network_continuous` — Evolutionary Network Simulation
Runs the full pipeline across multiple consecutive years, where each year's network evolves from the previous year.

Key arguments: `--sandbox_folder`, `--year` or `--start_year`/`--end_year`, `--methods`, `--personality_mode`

#### `supplier` — Single-entity Supplier Selection
Selects the best suppliers for a given target company and year using LLM-based personality modeling and scoring.

| Stage | Description |
|---|---|
| Stage 1 | Load all data (parquet) |
| Stage 2 | Build target company profile + supplier profiles |
| Stage 3 | Generate decision-maker personality via LLM |
| Stage 4 | Parallel supplier assessment |
| Stage 5 | Select top-k suppliers |
| Stage 6 | Evaluate selection accuracy |

Key arguments: `--entity_id`, `--year`, `--use_llm`, `--debug`, `--max_candidates`, `--top_k`

### Data Layout

```
supply_chain/
├── data/
│   └── factset/
│       ├── data/
│       │   ├── standard_entity.parquet
│       │   ├── company_factset.parquet
│       │   ├── factset_revere_relationship.parquet
│       │   ├── ff_usc_qf.parquet
│       │   ├── ff_int_qf.parquet
│       │   └── ...
│       └── factset_own/
│           └── own_basic.parquet
└── result/          # experiment outputs
```

### LLM Configuration (`.env`)

```bash
OPENAI_API_KEY=sk-...
BASE_URL=http://your-api-endpoint/v1/
# SUPPLY_CHAIN_MODEL=deepseek-chat    # optional override (default: MODEL env var)
```

---

## Unified LLM Configuration

All three modules share the same `.env` file at the project root:

```bash
# ── Shared credentials ──────────────────────────────────────────
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...          # same key as above on aggregation platforms
BASE_URL=http://35.220.164.252:3888/v1/
MODEL=USD-guiji/deepseek-v3

# ── AgentEconomy ────────────────────────────────────────────────
SIMPLE_MODEL=openai/gpt-4.1-nano
STRONG_MODEL=openai/gpt-3.5-turbo-16k
LLM_MAX_CONCURRENCY=200

# ── MarketSim (optional overrides) ──────────────────────────────
# MARKETSIM_THINK_MODEL=deepseek-reasoner
# MARKETSIM_GENERATE_MODEL=deepseek-chat

# ── Supply Chain (optional override) ────────────────────────────
# SUPPLY_CHAIN_MODEL=deepseek-chat
```

---

## Project Structure

```
AgentEconomy/
├── main.py                        # Unified entry point
├── config/
│   └── config_normal.yaml         # AgentEconomy config
├── agenteconomy/                  # Module 1: Macro simulation
│   ├── agent/                     #   household / firm / government / bank
│   ├── center/                    #   EconomicCenter / LaborMarket / ProductMarket
│   ├── simulation/                #   Simulator / Initializer
│   ├── llm/                       #   LLM router (litellm)
│   └── utils/                     #   plot_figures, logger, ...
├── marketsim/                     # Module 2: Stock market simulation
│   ├── run_marketsim.py           #   module entry point
│   ├── config_LLM.py              #   LLM config (reads .env)
│   ├── Kernel.py                  #   simulation kernel
│   ├── agent/                     #   trading agents
│   ├── Agent_FLLM/                #   LLM-based manager agents
│   └── config/                    #   experiment configs (rsmtry_LLM3, ...)
└── supply_chain/                  # Module 3: Supply chain research
    ├── run_supply_chain.py        #   module entry point
    ├── llm.py                     #   sync LLM bridge (reads .env)
    ├── code/
    │   ├── network/               #   static network reconstruction
    │   ├── network_continuous/    #   evolutionary network simulation
    │   └── supplier/              #   single-entity supplier selection
    ├── data/factset/              #   input data (parquet files)
    └── result/                    #   experiment outputs
```
