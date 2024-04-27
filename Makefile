format:
	poetry run ruff format .
	poetry run ruff check --fix-only .

lint:
	poetry run ruff check .
	poetry run ruff format --check .
	poetry run pyright sae_vis

test:
	poetry run pytest tests/unit

check-all:
	make format
	make lint
	make test
