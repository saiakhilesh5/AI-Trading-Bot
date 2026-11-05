# Autonomous Multi-Agent AI Trading Intelligence System

Gold (XAU/USD) · Silver (XAG/USD) · Bitcoin (BTC) · Ethereum (ETH)

A production-oriented, autonomous **paper-trading** platform that combines
quantitative analysis, statistical forecasting, real-time news intelligence,
macroeconomic interpretation and deterministic risk management.

> **Paper trading only.** Real-money execution is disabled and not implemented.
> No broker credentials are used or stored. Backtest results are never live results.

## Architecture (8 layers)

1. **Data foundation** — CCXT/Binance, CoinGecko, yfinance COMEX, GoldPrice.dev,
   FRED, RSS+GDELT news; point-in-time storage on TimescaleDB; health
   validation (gaps, staleness, duplicates, conflicting feeds, price jumps).
2. **Features** — technical, statistical, asset-specific, regime detection.
3. **Forecasting** — walk-forward benchmarking vs random-walk baselines,
   Diebold-Mariano testing, calibration, model registry (LightGBM, XGBoost,
   GARCH, HAR enabled; foundation models as disabled candidates).
4. **AI agents** — news intelligence, macro interpretation, decision reviewer
   (LangGraph; provider-agnostic LLM: OpenAI/Anthropic/Google/local).
   Reviewer can only **reduce** risk and fails closed when the LLM is down.
5. **Decision engine** — gates, interpretable signal combination, cost-aware
   thresholds, every NO TRADE logged with its reason.
6. **Risk engine** — final authority. Kill switches, drawdown ladder, daily
   loss limits, loss-streak cooldown, exposure caps (correlation-aware),
   volatility-targeted sizing.
7. **Paper execution** — realistic fills (spread, slippage, fees, partial
   fills), SL/TP brackets, portfolio accounting that survives restarts.
8. **Monitoring** — Streamlit dashboard, structured JSON logs, daily reports,
   LLM cost ledger, model drift refresh workflow.

## Quick start

```bash
docker compose up -d                    # TimescaleDB (5433) + Redis (6380)
copy .env.example .env                  # add FRED_API_KEY and LLM_API_KEY
python -m scripts.initialize_database
python -m scripts.download_historical_data
python -m workflows.run_forecasting     # benchmark + activate validated models
python -m scripts.run_paper_trading      # 24/7 supervisor (paper only)
streamlit run dashboard/app.py           # control centre
```

Full instructions: [OPERATIONS.md](OPERATIONS.md). Status and honest
benchmark results: [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md).

## Safety principles

- Quantitative evidence before AI interpretation; no fabricated data.
- Point-in-time integrity — no look-ahead, ever (walk-forward with gaps).
- Deterministic risk controls with final authority; LLMs cannot override.
- Untrusted external content is delimited and never followed as instructions.
- LLM outputs are schema-validated; malformed responses are rejected.
- Complete auditability: every decision, order, fill and cost is persisted.

## Tests

```bash
python -m pytest
```
