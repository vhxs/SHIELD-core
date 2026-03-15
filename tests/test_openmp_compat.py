"""
Regression test: pyOpenFHE must not segfault when libomp is pre-loaded
into the process before the extension is imported (as happens when torch
or other OpenMP-using libraries are imported first).

The check runs in a subprocess so the import order is guaranteed fresh,
independent of what other tests in the session have already imported.
Only meaningful on macOS; skipped elsewhere.
"""

import sys
import subprocess
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="OpenMP conflict only occurs on macOS"
)

_SCRIPT = """
import ctypes
import ctypes.util

# Simulate a library (e.g. torch) pre-loading libomp before pyOpenFHE is imported.
# libomp is keg-only on Homebrew and won't appear in standard library paths,
# so fall back to the well-known Homebrew prefix if find_library misses it.
libomp = ctypes.util.find_library("omp") or "/opt/homebrew/opt/libomp/lib/libomp.dylib"
ctypes.CDLL(libomp)

import numpy as np
from pyOpenFHE import CKKS
from pyOpenFHE import enums as pal

cc = CKKS.genCryptoContextCKKS(
    multiplicativeDepth=2,
    scalingFactorBits=50,
    batchSize=8,
)
cc.enable(pal.PKESchemeFeature.PKE)
cc.enable(pal.PKESchemeFeature.KEYSWITCH)
cc.enable(pal.PKESchemeFeature.LEVELEDSHE)

keys = cc.keyGen()
data = np.ones(8)
ct = cc.encrypt(keys.publicKey, data)
result = cc.decrypt(keys.secretKey, ct)[:8]
assert abs(result[0] - 1.0) < 1e-4, f"unexpected result: {result}"
"""


def test_he_operation_with_libomp_preloaded():
    """pyOpenFHE must work correctly when libomp is already loaded in the process."""
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"Process exited with code {proc.returncode}\n"
        f"stdout: {proc.stdout.decode()}\n"
        f"stderr: {proc.stderr.decode()}"
    )
