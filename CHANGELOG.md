# Changelog

All notable changes to H4wkQuant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-04-01

### Added
- Initial release of H4wkQuant
- 10 mathematical models (Bayesian, Kalman, Kelly, Stoikov, Monte Carlo, etc.)
- 5 microservices architecture
- 4 trading strategies (stat_arb, funding_arb, momentum_div, cross_exchange)
- Web panel with Vue.js + FastAPI
- Docker Compose setup
- Prometheus + Grafana monitoring
- Risk management with kill switch
- Paper and Live trading modes

### Features
- Real-time Binance Futures data collection
- Z-score based statistical arbitrage
- Funding rate arbitrage
- Kelly Criterion position sizing
- Multi-layer risk controls
- Telegram notifications
- JWT authentication
- TimescaleDB for time-series data
