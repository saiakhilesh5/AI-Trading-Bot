# Development Status

Autonomous multi-agent AI trading intelligence system — Gold, Silver, Bitcoin, Ethereum.
**Paper trading only. Real-money execution is disabled and not implemented.**

Last updated: 2026-11-05

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Project foundation and configuration | Done |
| 2 | Data collection and validation | Done |
| 3 | Database and historical data management | Done |
| 4 | Feature engineering | Done |
| 5 | Market-regime detection | Done |
| 6 | Forecasting benchmarks | Done |
| 7 | News intelligence (agent) | Done |
| 8 | Macroeconomic intelligence (agent) | Done |
| 9 | Agent coordination (LangGraph) | Done |
| 10 | Decision engine | Done |
| 11 | Risk management | Done |
| 12 | Paper execution | Done |
| 13 | Portfolio accounting | Done |
| 14 | Dashboard | Done |
| 15 | Backtesting and validation | Done |
| 16 | Continuous local operation | Done |
| 17 | Cloud migration readiness | By design (env-driven config, no local paths in code) |

## What is implemented

**Layer 1 — Data Foundation.** CCXT (Binance, paginated), CoinGecko fallback,
yfinance COMEX futures for metals with chunked fetching, GoldPrice.dev spot
cross-check, FRED macro, RSS + GDELT news. Session-aware data-health validation
(duplicates, gaps, staleness, price jumps, invalid values, conflicting feeds,
provider failures, latency). Point-in-time storage: event / publication /
ingestion timestamps preserved; `as_of` reads filter by ingestion time;
first-ingested version wins (no history rewrites). TimescaleDB hypertables for
candles/derivatives/macro/news/events/forecasts.

**Layer 2 — Features.** Deterministic technical features (RSI, MACD, EMA/SMA,
VWAP, ATR, Bollinger, momentum, ROC, support/resistance, breakouts, volume
anomaly, trend strength), statistical features (realized/historical volatility,
skew, kurtosis, drawdown, vol clustering, autocorrelation, rolling correlation),
asset-specific features (gold-silver ratio, yields, DXY), and a
volatility/trend regime classifier.

**Layer 3 — Forecasting.** Walk-forward evaluation with a train/eval gap
(no leakage), Diebold-Mariano test vs the random-walk baseline, interval
coverage/Brier calibration. Candidates: LightGBM, XGBoost, GARCH, HAR
(foundation models Chronos-2/TimesFM/Moirai/PatchTST/iTransformer are wired as
config entries but disabled — see Outstanding Issues). Model registry with
registered/active/shadow/disabled statuses; only models that beat the random
walk with >= 100 out-of-sample predictions are adopted and promoted.

**Layer 4 — Agents.** Provider-agnostic LLM client (OpenAI / Anthropic /
Google / any OpenAI-compatible local endpoint via LLM_PROVIDER+LLM_MODEL).
News Intelligence Agent (deterministic dedup/staleness pre-filter, LLM
extraction into strict Pydantic schemas, output sanitization). Macro Agent
(point-in-time observations only). Decision Reviewer Agent (may only reduce
risk; fails CLOSED when the LLM is unavailable). LangGraph orchestration.
All external content wrapped in UNTRUSTED markers; structured outputs are
validated and malformed responses are rejected, never patched. Persistent
LLM cost ledger with a monthly budget guard.

**Layer 5 — Decision Engine.** Trading gates (data health, regime suitability,
forecast plausibility, upcoming releases, news conflicts), interpretable
signal combination (forecast + technical + regime votes), round-trip cost
estimation (spread + fees + slippage + funding) that must be cleared by the
expected return, reviewer verdict, then risk engine. Every NO TRADE is
recorded with its reason.

**Layer 6 — Risk Engine (final authority).** Kill switches (global and
per-asset, Redis-backed), drawdown ladder (4% reduce / 8% halt / 12% emergency
shutdown with manual-restart requirement), daily loss limit, loss-streak
cooldown, order frequency limits, max open positions, duplicate-direction
block, stop/TP validity, min reward-risk ratio, per-asset and correlated-group
exposure caps with measured-correlation relief, volatility-targeted sizing.
The reviewer can only shrink position size, never grow it.

**Layer 7 — Paper Execution.** Simulated broker only (no real-order code path
exists). Spread, volatility-scaled slippage, fees, partial fills for oversized
orders, limit-order reachability checks, SL/TP bracket processing on each
mark. Portfolio accounting persisted in the DB (cash, peak equity, daily P&L,
consecutive losses) so restarts lose nothing. Reconciliation verifies
position/fill/decision invariants and flags anomalies.

**Layer 8 — Monitoring & Dashboard.** Streamlit control centre with 8 pages
(market overview, news intelligence, forecasting, agent monitoring, decisions,
paper portfolio, risk monitoring, model performance). Structured JSON logs
(Loguru), daily report generator, LLM cost ledger, model refresh workflow
(re-benchmark + promote/demote), scheduler with heartbeats and startup
recovery sequence (DB retry, Redis, kill-switch check, reconciliation).

**Validation.** 118 tests covering contracts, health validation, events,
repository, agents (degradation paths, untrusted-content handling, schema
rejection), risk engine (all 10+ rejection paths), decision engine (gate and
reviewer interactions), execution (fills, brackets, insufficient balance),
restart recovery (portfolio survives, replayed decision rejected, no duplicate
fills), and the 8-strategy backtest engine (no-look-ahead test included).

## Benchmark results (walk-forward, real data, 2026-11-05)

Daily horizon: only GARCH beat the random-walk baseline for XAUUSD and XAGUSD
(MAE lower, DM test), and for ETHUSD on 1h. LightGBM/XGBoost/HAR and all other
candidates did NOT beat the random walk. Baseline directional accuracy across
assets/horizons: ~0.47-0.54. This is an honest result: markets are close to
efficient at these horizons and the system does not fabricate an edge.

Strategy comparison (BTCUSD 1d, out-of-sample walk-forward means): buy-and-hold
(+4.76%/window) and trend-following (+2.87%/window) outperformed the multi-signal
configurations on this bull-market sample. Full results in
`reports/strategy_comparison_*.json`. **These are backtest results, not live
performance.**

## Outstanding issues

1. **FRED API key not configured** — macro ingestion is skipped gracefully.
   Add `FRED_API_KEY` to `.env` to enable. Free at fred.stlouisfed.org.
2. **LLM not configured** — agents degrade to empty-but-valid analyses; the
   reviewer FAILS CLOSED, so no trades can pass review until an LLM key is
   configured (`LLM_API_KEY`, optionally `LLM_PROVIDER`/`LLM_MODEL`). This is
   intentional: an unconfigured system must not trade on unreviewed signals.
3. **Foundation forecasting models disabled** — Chronos-2, TimesFM, Moirai,
   PatchTST, iTransformer are listed in `config/model_config.yaml` as disabled
   candidates. They require large downloads and GPU inference; enabling them
   means adding their loaders to `forecasting/candidate_models()`.
4. **Metals intraday depth** — yfinance intraday history is capped (~60 days
   for 1h). Metals 1h/4h windows are sufficient for current walk-forward
   minimums but shallower than crypto (140 days).
5. **Alembic not wired** — schema is created via `create_all` + drop/recreate
   for changed tables. Introduce Alembic before first cloud deployment.
6. **News event persistence** — extracted events are stored in
   `agent_outputs`; a dedicated `market_events` table is defined but the
   pipeline writes agent JSON. Consider normalizing if event queries grow.
7. **CoinGecko fallback for crypto candles is daily-only** — intraday crypto
   relies on Binance only (no free intraday fallback identified).

## Verified end-to-end (2026-11-05)

- Full supervisor cycle (`--once`): live candle ingestion for all 4 assets,
  real news ingestion (RSS+GDELT), agent graph runs, daily report, graceful
  shutdown; zero scheduler task failures.
- Model refresh workflow re-benchmarked all asset/horizon pairs and promoted
  GARCH (XAUUSD 1d, XAGUSD 1d) to `active` — these are the only models that
  beat the random-walk baseline.
- Live decision cycle used the active GARCH models, persisted forecast
  records, and correctly produced NO TRADE after the cost gate (net expected
  return negative after spread/fees/slippage/funding) — the full gates ->
  forecast -> signal combination -> cost check path executed against real data.
- Assets without a validated model (BTC, ETH at these horizons) honestly
  report `no_validated_model` instead of fabricating a forecast.
- Health check: all infrastructure and data-freshness checks pass.
- Dashboard serves all 8 pages.
