# H4wkQuant

**Mathematical Arbitrage Trading System for Binance Futures**

Designed & developed by **H4wk**

---

## Overview

H4wkQuant is a microservice-based quantitative trading system that runs statistical arbitrage, funding rate arbitrage, and momentum divergence strategies on Binance Futures. It features 10 mathematical models, real-time signal generation, multi-layer risk management, and a web-based control panel.

## Architecture

```
                    ┌─────────────────┐
                    │  data_collector  │  Binance/Bybit WS → Redis
                    └────────┬────────┘
                             │ ticks, orderbook, funding
                             ▼
                    ┌─────────────────┐
                    │  spread_engine   │  Z-score, Bayesian, Kelly → Signals
                    └────────┬────────┘
                             │ arb.signal
                             ▼
                    ┌─────────────────┐
                    │  risk_manager    │  Leverage, drawdown, kill switch
                    └────────┬────────┘
                             │ arb.approved
                             ▼
                    ┌─────────────────┐
                    │    executor      │  Stoikov pricing → Binance orders
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ watchdog  │  │  panel   │  │ postgres │
        │ (health)  │  │ (web UI) │  │ (trades) │
        └──────────┘  └──────────┘  └──────────┘
```

All inter-service communication goes through **Redis pub/sub** — no direct HTTP between services.

## Services

| Service | Description | Port |
|---------|-------------|------|
| `data_collector` | Real-time price ingestion via Binance WebSocket (trades, orderbook L20, mark price, funding rates) | 9101 |
| `spread_engine` | Signal generation pipeline: SpreadModel → RegimeDetector → PortfolioOptimizer → BayesianModel → EdgeModel → KellyModel | 9102 |
| `risk_manager` | Position limits, leverage caps, daily/weekly drawdown checks, kill switch enforcement | 9103 |
| `executor` | Order management with Stoikov limit pricing, paper/live mode, position cooldown (60s) | 9104 |
| `watchdog` | Service health monitoring (30s intervals), Telegram alerts, kill switch management | 9105 |
| `panel` | Vue.js + FastAPI web dashboard with real-time WebSocket updates, 40+ REST endpoints | 8180 |

## Mathematical Models

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| **SpreadModel** | Z-score + ADF cointegration test | Core stat-arb signal: ratio, spread, z-score, half-life |
| **BayesianModel** | Posterior update: P(H\|D) = P(D\|H) × P(H) / P(D) | Fair value estimation from orderbook imbalance, volume, funding, OI |
| **KalmanFilter** | 1D state-space model | Smooth noisy observations (half-life, hedge ratio) |
| **KellyCriterion** | f* = (bp - q) / b | Optimal position sizing (Quarter Kelly, conservative) |
| **EdgeModel** | EV_net = q - p - c | Net expected value after ALL costs (commission, slippage, funding) |
| **StoikovModel** | Avellaneda-Stoikov (2008) | Inventory-aware limit order placement |
| **MonteCarloModel** | Wealth process simulation (1000+ paths) | Strategy validation: Sharpe, max drawdown, ruin probability |
| **RegimeDetector** | BTC volatility percentile classification | Market regime gating (blocks trades in HIGH/EXTREME) |
| **PortfolioOptimizer** | Rolling correlation matrix | Prevents opening correlated pairs (max correlation 0.7) |
| **TimeframeAggregator** | OHLC candle construction | Multi-timeframe analysis (5m/15m cointegration check) |

## Strategies

| Strategy | Type | Entry Logic |
|----------|------|-------------|
| **stat_arb** | Statistical arbitrage | Z-score > 3.0 + cointegration confirmed + Bayesian posterior + positive edge + Kelly sizing |
| **funding_arb** | Funding rate arbitrage | \|Funding\| > 0.05% per 8h OR annualized > 10%. Short funded asset, hedge with pair |
| **momentum_div** | Momentum divergence | BTC moves > 1%, alt lags. Enter alt in BTC direction, exit within 30s |
| **cross_exchange** | Cross-exchange arbitrage | Binance vs Bybit price diff > 0.15% after fees |

## Risk Controls

Five independent layers of risk management:

| Layer | Controls |
|-------|----------|
| **Spread Engine** | Regime detection, portfolio correlation, cointegration validation |
| **Risk Manager** | Max leverage 3x, max 6 positions, daily loss 2%, weekly 5%, drawdown 10% |
| **Executor** | Stoikov conservative pricing, position cooldown 60s, order tracking |
| **Watchdog** | Service health monitoring, emergency alerts |
| **Kill Switch** | Immediate halt of all trading activity |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Web Framework | FastAPI + Uvicorn |
| Frontend | Vue.js 3 + Tailwind CSS + Chart.js |
| Database | TimescaleDB (PostgreSQL 16) |
| Cache/Bus | Redis 7 (pub/sub + state) |
| Math | NumPy, Pandas, Statsmodels |
| Containers | Docker Compose |
| Monitoring | Prometheus + Grafana |
| Security | JWT (HMAC-SHA256), PBKDF2, Fernet encryption for API keys |

## Database Schema

| Table | Purpose |
|-------|---------|
| `arb_trades` | Paired 2-leg arbitrage trade records with PnL |
| `spread_snapshots` | TimescaleDB hypertable, 30-day retention, z-score history |
| `daily_stats` | Daily PnL aggregation, Sharpe ratio |
| `account_snapshots` | Balance and position history |

## Setup

```bash
# Clone and configure
cp .env.example .env
# Edit .env with your API keys

# Start (development)
make up

# Run tests
make test

# Backtest
make backtest

# Pair screening
make screen

# Pre-flight check before going live
make preflight
```

## Production Deployment

Deploys to a remote server via Tailscale VPN:

```bash
# Full deploy (rsync + build + start)
make deploy

# Code only (no rebuild)
make deploy-code

# Service management
make prod-up
make prod-down
make prod-logs
make prod-status
```

Panel is accessible only via Tailscale IP on port 8180.

## Scripts

| Script | Purpose |
|--------|---------|
| `backtest_pairs.py` | Historical backtest on configured pairs |
| `backtest_enhanced.py` | Walk-forward backtest with equity curve |
| `calibrate_models.py` | Parameter optimization from live market data |
| `pair_screener.py` | Scans 500+ Binance pairs for cointegration |
| `preflight_check.py` | Safety validation before live trading |
| `validate_strategy.py` | Monte Carlo strategy viability test |

## Configuration

All configuration via environment variables (`.env`) with runtime overrides through Redis (adjustable from panel):

- **Trading pairs**: Configurable (default: BTCUSDT/ETHUSDT, SOLUSDT/ETHUSDT, BNBUSDT/ETHUSDT)
- **Z-score thresholds**: Entry 3.0, exit 0.8
- **Kelly fraction**: 0.25 (Quarter Kelly)
- **Monte Carlo**: 1000 simulations, min Sharpe 1.0, max ruin probability 1%
- **Trading mode**: Paper (simulated) or Live (real orders)

## Project Structure

```
H4wkQuant/
├── models/              # 10 mathematical models
│   ├── bayesian.py
│   ├── kalman.py
│   ├── kelly.py
│   ├── stoikov.py
│   ├── montecarlo.py
│   ├── regime.py
│   ├── spread.py
│   ├── edge.py
│   ├── portfolio.py
│   ├── timeframe.py
│   └── tests/
├── strategies/          # 4 trading strategies
│   ├── stat_arb.py
│   ├── funding_arb.py
│   ├── momentum_div.py
│   └── cross_exchange.py
├── services/            # 5 microservices
│   ├── data_collector/
│   ├── spread_engine/
│   ├── risk_manager/
│   ├── executor/
│   └── watchdog/
├── panel/               # Web dashboard
│   ├── backend/         # FastAPI (40+ endpoints, WebSocket)
│   └── frontend/        # Vue.js SPA
├── shared/              # Shared libraries
│   ├── clients/         # Binance, Bybit, Telegram clients
│   ├── config/          # Pydantic settings
│   ├── database/        # SQLAlchemy ORM + TimescaleDB
│   ├── schemas/         # Pydantic models
│   └── utils/           # Circuit breaker, rate limiter, retry, metrics
├── scripts/             # Backtest, screening, calibration tools
├── infra/docker/        # Docker Compose, Dockerfiles, Grafana, Prometheus
└── docs/                # Project documentation
```

---

**H4wkQuant** — by H4wk
