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

profile-memory-unit:
	poetry run pytest --memray tests/unit

profile-speed-unit:
	poetry run py.test tests/unit --profile-svg -k "test_SaeVisData_create_results_look_reasonable[Default]"
	open prof/combined.svg
