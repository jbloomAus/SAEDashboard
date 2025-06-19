format:
	poetry run black .
	poetry run isort .

lint:
	poetry run flake8 .
	poetry run black --check .
	poetry run isort --check-only --diff .

check-type:
	poetry run pyright .

test:
	poetry run pytest --cov=sae_dashboard --cov-report=term-missing tests/unit tests/acceptance

check-ci:
	make format
	make lint
	make check-type
	make test

profile-memory-unit:
	poetry run pytest --memray tests/unit

profile-speed-unit:
	poetry run py.test tests/unit --profile-svg -k "test_SaeVisData_create_results_look_reasonable[Default]"
	open prof/combined.svg
