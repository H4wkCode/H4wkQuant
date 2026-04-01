-- H4wkQuant - PostgreSQL Initialization
-- TimescaleDB hypertables for time-series data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Arb Trades table
CREATE TABLE IF NOT EXISTS arb_trades (
    id SERIAL PRIMARY KEY,
    pair_id VARCHAR(50) NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    leg_a_symbol VARCHAR(20) NOT NULL,
    leg_a_side VARCHAR(10) NOT NULL,
    leg_a_entry_price DOUBLE PRECISION,
    leg_a_exit_price DOUBLE PRECISION,
    leg_a_quantity DOUBLE PRECISION,
    leg_b_symbol VARCHAR(20) NOT NULL,
    leg_b_side VARCHAR(10) NOT NULL,
    leg_b_entry_price DOUBLE PRECISION,
    leg_b_exit_price DOUBLE PRECISION,
    leg_b_quantity DOUBLE PRECISION,
    leverage INTEGER DEFAULT 3,
    entry_zscore DOUBLE PRECISION DEFAULT 0,
    exit_zscore DOUBLE PRECISION DEFAULT 0,
    combined_pnl DOUBLE PRECISION DEFAULT 0,
    total_commission DOUBLE PRECISION DEFAULT 0,
    exit_reason VARCHAR(50),
    trading_mode VARCHAR(10) DEFAULT 'paper',
    entry_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    exit_time TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_arb_trades_pair_time ON arb_trades(pair_id, entry_time);
CREATE INDEX IF NOT EXISTS idx_arb_trades_strategy ON arb_trades(strategy);

-- Spread Snapshots (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS spread_snapshots (
    time TIMESTAMPTZ NOT NULL,
    pair_id VARCHAR(50) NOT NULL,
    ratio DOUBLE PRECISION,
    spread DOUBLE PRECISION,
    zscore DOUBLE PRECISION,
    mean DOUBLE PRECISION,
    std DOUBLE PRECISION,
    half_life DOUBLE PRECISION,
    coint_pvalue DOUBLE PRECISION
);

SELECT create_hypertable('spread_snapshots', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_spread_pair ON spread_snapshots(pair_id, time DESC);

-- Daily Stats
CREATE TABLE IF NOT EXISTS daily_stats (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    realized_pnl DOUBLE PRECISION DEFAULT 0,
    trade_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    sharpe_ratio DOUBLE PRECISION DEFAULT 0,
    balance_start DOUBLE PRECISION DEFAULT 0,
    balance_end DOUBLE PRECISION DEFAULT 0,
    best_trade DOUBLE PRECISION DEFAULT 0,
    worst_trade DOUBLE PRECISION DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Account Snapshots
CREATE TABLE IF NOT EXISTS account_snapshots (
    id SERIAL PRIMARY KEY,
    total_balance DOUBLE PRECISION,
    available_balance DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION DEFAULT 0,
    realized_pnl_today DOUBLE PRECISION DEFAULT 0,
    open_positions INTEGER DEFAULT 0,
    total_leverage DOUBLE PRECISION DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Retention policy: keep spread snapshots for 30 days
SELECT add_retention_policy('spread_snapshots', INTERVAL '30 days', if_not_exists => TRUE);
