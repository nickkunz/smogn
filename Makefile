# Run tests
.PHONY: test

test:
	pytest
	
# Setup dev dependencies
.PHONY: dev

dev:
	pip -e .[dev]