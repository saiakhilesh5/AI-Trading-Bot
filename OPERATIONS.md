# Operations Guide

Paper-trading system. Real-money execution is **not implemented and disabled**.

## 1. Prerequisites

- Python 3.10+ (venv in `.venv`)
- Docker Desktop (TimescaleDB + Redis)

## 2. Start the infrastructure

```bash
docker compose up -d
```

- TimescaleDB on `localhost:5433` (user `aits`, password `aits_local_dev`)
- Redis on `localhost:6380`

## 3. Configure

```bash
copy .env.example .env   # then edit
```

| Variable | Needed for | Notes |
|---|---|---|
| `FRED_API_KEY` | Macro data + macro agent | Free; without it macro ingestion is skipped |
| `LLM_API_KEY` | News/macro/reviewer agents | Without it the reviewer fails closed => no trades |
| `LLM_PROVIDER` / `LLM_MODEL` | LLM choice | openai / anthropic / google / local (set `LLM_BASE_URL` for local) |
| `LLM_MONTHLY_BUDGET_USD` | Cost guard | Default 10 USD/month, enforced from the cost ledger |
| `PAPER_STARTING_CAPITAL_USD` | Portfolio | Default 100,000 USD |

## 4. Initialize and backfill

```bash
python -m scripts.initialize_database
python -m scripts.download_historical_data        # all assets, ~minutes
```

## 5. Benchmark and activate models

```bash
python -m scripts.run_backtest --asset BTCUSD --interval 1d --horizon 1d
python -m workflows.run_forecasting               # re-benchmark + promote winners
python -m scripts.run_strategy_comparison --asset BTCUSD --interval 1d
```

Only models that beat the random-walk baseline are promoted to `active`.
The trading loop will NOT trade an asset with no active model (reason:
`no_validated_model` in the decision log).

## 6. Run continuously (24/7)

```bash
python -m scripts.run_paper_trading
```

- Candles + trade cycle every 15 minutes; news and agents every 30 minutes;
  macro every 6 hours; model refresh and daily report once a day.
- One-shot for testing: `python -m scripts.run_paper_trading --once`
- Graceful shutdown: Ctrl+C (finishes the current task, then exits).

**For genuine 24/7 operation**: disable Windows sleep, keep the machine
powered and connected, and consider running the supervisor under a service
manager (NSSM / Task Scheduler) with auto-restart.

## 7. Dashboard

```bash
streamlit run dashboard/app.py
```

8 pages: market overview, news intelligence, forecasting, agent monitoring,
trading decisions, paper portfolio, risk monitoring, model performance.

## 8. Recovery after restart / crash

The system is designed to restart safely:

- Portfolio state, positions, fills, orders, decisions live in TimescaleDB.
- Startup runs the recovery sequence: DB retry (up to 12x) -> Redis check ->
  kill-switch check -> position reconciliation -> heartbeat.
- Replaying a decision is rejected (duplicate-order protection).
- If the emergency drawdown shutdown fired, the supervisor refuses to start
  new trades until the kill switch is manually cleared (see below).

## 9. Emergency procedures

```python
# from a python shell in the project venv
from risk.kill_switch import KillSwitch
ks = KillSwitch()
ks.trigger("reason")            # global halt; ks.trigger("reason", asset="XAUUSD") for one asset
ks.clear()                      # manual restart after investigating
ks.clear(asset="XAUUSD")
```

Kill-switch state is Redis-backed and survives restarts. The drawdown ladder
(4% reduce / 8% halt / 12% emergency shutdown) is configured in
`config/risk_config.yaml`.

## 10. Health check

```bash
python -m scripts.health_check
```

## 11. Logs and reports

- `logs/app_YYYY-MM-DD.log` — structured JSON (Loguru)
- `reports/daily/YYYY-MM-DD.md` — daily portfolio/decision/cost summary
- `reports/benchmark_latest.json`, `reports/strategy_comparison_*.json`
- LLM spend: dashboard page 04, or the `cost_ledger` table

## 12. Rules that must never be relaxed

1. Paper trading only. No real-money execution pathway may be added without a
   separate, explicit authorization process.
2. The risk engine has final authority. No LLM or agent may override it.
3. The reviewer agent can only reduce risk, never increase it.
4. Never fabricate data, forecasts, agent responses, or performance numbers.
5. Backtest results are never live results.
