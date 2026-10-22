.PHONY: setup train monitor serve loadtest test docker web copilot-eval all

setup:      ## install package + dev tools
	pip install -r requirements-dev.txt && pip install -e .

train:      ## offline experiment: rules vs LR vs LightGBM, saves model v1
	python scripts/run_experiment.py

monitor:    ## 20-week production simulation with drift + retraining policies
	python scripts/run_monitoring.py

serve:      ## scoring API on :8000 (docs at /docs); loads .env if present
	set -a; [ -f .env ] && . ./.env; set +a; uvicorn fraudguard.api:app --port 8000

loadtest:   ## latency benchmark: pandas scorer vs numpy fast path
	python scripts/load_test.py

test:
	pytest -v

docker:
	docker build -t fraud-drift-guard . && docker run --rm -p 8000:8000 fraud-drift-guard

copilot-eval: ## LLM copilot eval (uses cached responses; needs ANTHROPIC_API_KEY only for new calls)
	python scripts/eval_copilot.py

web:        ## export model to docs/model.json for the GitHub Pages demo
	python scripts/export_web_model.py

all: train monitor loadtest web test
