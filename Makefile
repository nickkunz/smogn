# Run tests
.PHONY: test

test:
	pytest
	
# Setup dev dependencies
.PHONY: dev

dev:
	pip install -r ./requirements.txt \
	pip -e .[dev]