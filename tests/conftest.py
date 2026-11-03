"""Test configuration: keep every run-time output out of the working tree and make the package importable."""

import os
import tempfile

# Must be set before `arretrieval.*.paths` is imported (they create their output directories at import time).
os.environ.setdefault("ARR_RUNS_DIR", tempfile.mkdtemp(prefix="arr-test-runs-"))
