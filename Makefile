format:
	poetry run ruff format .
	poetry run ruff check --fix-only .

check-format:
	poetry run ruff check .
	poetry run ruff format --check .

check-type:
	poetry run pyright .

test:
	poetry run pytest --cov=sae_dashboard --cov-report=term-missing tests/unit

check-ci:
	make check-format
	make check-type
	make test

profile-memory-unit:
	poetry run pytest --memray tests/unit

profile-speed-unit:
	poetry run py.test tests/unit --profile-svg -k "test_SaeVisData_create_results_look_reasonable[Default]"
	open prof/combined.svg
