PYTHON ?= .venv/bin/python
.PHONY: test
test:
	JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 $(PYTHON) -m pytest -q

RECOVERY_OUTPUT ?= work/p1-repeat
.PHONY: recovery
recovery:
	JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 $(PYTHON) tests/bench/recovery.py --output "$(RECOVERY_OUTPUT)"
