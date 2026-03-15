"""Tests for CNN.fhe_gelu.

fhe_gelu evaluates a Chebyshev approximation of gelu(x * bound) on [-1, 1].
Raises RuntimeError if getTowersRemaining() is too low for the requested degree.
"""

import math
import numpy as np
import pytest

pytest.importorskip("pyOpenFHE")

from conftest import CKKS_ATOL

pytestmark = pytest.mark.slow

MTX_SIZE = 4
BATCH_SIZE = MTX_SIZE * MTX_SIZE


def _gelu(x: float) -> float:
    """Exact GELU: x * Φ(x) where Φ is the standard normal CDF."""
    return x * 0.5 * math.erfc(-x * math.sqrt(0.5))


def _gelu_scaled(x: float, bound: float) -> float:
    """fhe_gelu approximates gelu(x * bound) on the domain [-1, 1]."""
    return _gelu(x * bound)


class TestFheGelu:
    # -- smoke --

    def test_returns_list(self, cnn_cc_keys, cnn_shards):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        result = CNN.fhe_gelu(cnn_shards, 3, 5.0)
        assert isinstance(result, list)
        assert all(isinstance(c, CKKSCiphertext) for c in result)

    def test_output_decryptable(self, cnn_cc_keys, cnn_shards):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.fhe_gelu(cnn_shards, 3, 5.0)
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    # -- correctness --

    def test_insufficient_towers_raises(self):
        """fhe_gelu raises RuntimeError when too few multiplicative levels remain."""
        from pyOpenFHE import CKKS, enums as pal
        from pyOpenFHE.CKKS import CNN
        cc = CKKS.genCryptoContextCKKS(
            multiplicativeDepth=3, scalingFactorBits=59, batchSize=BATCH_SIZE
        )
        for feat in (
            pal.PKESchemeFeature.PKE,
            pal.PKESchemeFeature.KEYSWITCH,
            pal.PKESchemeFeature.LEVELEDSHE,
        ):
            cc.enable(feat)
        keys = cc.keyGen()
        ct = cc.encrypt(keys.publicKey, np.zeros(BATCH_SIZE, dtype=np.float64))
        with pytest.raises(RuntimeError):
            CNN.fhe_gelu([ct], 27, 1.0)

    def test_approximates_gelu_on_unit_interval(self, cnn_cc_keys):
        """With degree=27 and bound=1, fhe_gelu ≈ gelu(x) for x ∈ [-1, 1]."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        # Values well within [-1, 1] so the Chebyshev approximation is accurate.
        plaintext = np.array(
            [0.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.1,
             -0.1, 0.9, -0.9, 0.0,   0.3,  -0.3, 0.6, -0.6],
            dtype=np.float64,
        )
        ct = cc.encrypt(keys.publicKey, plaintext)
        result = CNN.fhe_gelu([ct], 27, 1.0)
        decrypted = np.array(cc.decrypt(keys.secretKey, result[0]))
        expected = np.array([_gelu_scaled(x, bound=1.0) for x in plaintext])
        # Tolerance is loose to allow for Chebyshev approximation error at degree=27
        # plus CKKS noise.
        np.testing.assert_allclose(decrypted[:BATCH_SIZE], expected, atol=1e-3)
