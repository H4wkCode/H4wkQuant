.PHONY: up down build logs test deploy backtest validate calibrate screen backtest-enhanced preflight go-live

COMPOSE_DEV = docker compose -f infra/docker/docker-compose.yml
COMPOSE_PROD = docker compose -f infra/docker/docker-compose.prod.yml
SSH = ssh user@your-server-ip
REMOTE_DIR = /home/user/h4wkquant

# ============================================================
# Docker (Local Development)
# ============================================================

up:
	cd infra/docker && docker compose up -d

down:
	cd infra/docker && docker compose down

build:
	cd infra/docker && docker compose build --no-cache

logs:
	cd infra/docker && docker compose logs -f --tail=100

restart:
	cd infra/docker && docker compose restart $(SERVICE)

# ============================================================
# Scripts & Testing
# ============================================================

test:
	PYTHONPATH=. pytest models/tests/ -v

backtest:
	PYTHONPATH=. python scripts/backtest_pairs.py

validate:
	PYTHONPATH=. python scripts/validate_strategy.py

calibrate:
	PYTHONPATH=. python scripts/calibrate_models.py

screen:
	PYTHONPATH=. python scripts/pair_screener.py

backtest-enhanced:
	PYTHONPATH=. python scripts/backtest_enhanced.py

preflight:
	PYTHONPATH=. python scripts/preflight_check.py

go-live:
	@echo "Running pre-flight checks..."
	PYTHONPATH=. python scripts/preflight_check.py && \
	echo "Pre-flight passed. Switching to LIVE mode..." && \
	sed -i 's/TRADING_MODE=paper/TRADING_MODE=live/' .env && \
	echo "TRADING_MODE set to live in .env" && \
	echo "Run 'make prod-up' or 'make deploy' to start live trading"

# ============================================================
# Production Deploy (Tailscale server)
# ============================================================

deploy: ## Deploy code to server and start
	rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'data/' \
		--exclude 'logs/' --exclude '~' \
		. $(SSH):$(REMOTE_DIR)/
	$(SSH) "cd $(REMOTE_DIR) && docker compose -f infra/docker/docker-compose.prod.yml build --no-cache && docker compose -f infra/docker/docker-compose.prod.yml up -d --force-recreate"

deploy-code: ## Send code only (no rebuild)
	rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'data/' \
		--exclude 'logs/' --exclude '~' \
		. $(SSH):$(REMOTE_DIR)/

prod-up: ## Start services on server
	$(SSH) "cd $(REMOTE_DIR) && docker compose -f infra/docker/docker-compose.prod.yml up -d"

prod-down: ## Stop services on server
	$(SSH) "cd $(REMOTE_DIR) && docker compose -f infra/docker/docker-compose.prod.yml down"

prod-restart: ## Restart services on server
	$(SSH) "cd $(REMOTE_DIR) && docker compose -f infra/docker/docker-compose.prod.yml down && docker compose -f infra/docker/docker-compose.prod.yml up -d"

prod-build: ## Rebuild on server
	$(SSH) "cd $(REMOTE_DIR) && docker compose -f infra/docker/docker-compose.prod.yml build --no-cache"

prod-logs: ## Show server logs
	$(SSH) "cd $(REMOTE_DIR) && docker compose -f infra/docker/docker-compose.prod.yml logs -f --tail=100"

prod-status: ## Show service status on server
	$(SSH) "cd $(REMOTE_DIR) && docker compose -f infra/docker/docker-compose.prod.yml ps"

prod-redis: ## Open Redis CLI on server
	$(SSH) "cd $(REMOTE_DIR) && docker compose -f infra/docker/docker-compose.prod.yml exec redis redis-cli -n 1"

# ============================================================
# Local Status
# ============================================================

status:
	@echo "=== Data Collector ===" && curl -s http://localhost:9101/health | python -m json.tool 2>/dev/null || echo "DOWN"
	@echo "=== Spread Engine ===" && curl -s http://localhost:9102/health | python -m json.tool 2>/dev/null || echo "DOWN"
	@echo "=== Risk Manager ===" && curl -s http://localhost:9103/health | python -m json.tool 2>/dev/null || echo "DOWN"
	@echo "=== Executor ===" && curl -s http://localhost:9104/health | python -m json.tool 2>/dev/null || echo "DOWN"
	@echo "=== Watchdog ===" && curl -s http://localhost:9105/health | python -m json.tool 2>/dev/null || echo "DOWN"

help: ## Help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
