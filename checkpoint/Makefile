.PHONY: all build clean install test

all: build

build:
	@cd src/native && $(MAKE)
	@cd src/native && $(MAKE) install

clean:
	@cd src/native && $(MAKE) clean
	@rm -rf build/ dist/ *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true

install: build
	@pip install -e .

test: build
	@pytest tests/ -v

rebuild: clean build

