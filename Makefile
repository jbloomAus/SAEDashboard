format:
	poetry run ruff format .
	poetry run ruff check --fix-only .

lint:
	poetry run ruff check .
	poetry run ruff format --check .
	poetry run pyright sae_dashboard

test:
	poetry run pytest tests/unit

check-ci:
	make format
	make lint
	make test

profile-memory-unit:
	poetry run pytest --memray tests/unit

profile-speed-unit:
	poetry run py.test tests/unit --profile-svg -k "test_SaeVisData_create_results_look_reasonable[Default]"
	open prof/combined.svg
