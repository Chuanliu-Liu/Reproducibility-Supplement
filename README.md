# Reproducibility Supplement

This archive is the minimal local release of the disclosure engine. It
contains the complete src/disclosure implementation, self-contained regression
tests, and one deterministic smoke example.

## Quick start

    python3 -m pip install -r requirements.txt
    python3 -m compileall -q src tests
    PYTHONPATH=src python3 examples/run_smoke.py
    PYTHONPATH=src python3 -m pytest -q

The test suite uses synthetic fixtures and does not access the network. The
smoke output is deterministic up to the floating-point implementation.
