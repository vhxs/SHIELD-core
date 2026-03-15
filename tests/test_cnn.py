"""
Smoke tests for CKKS CNN operations.

These tests verify that the CNN module functions are callable and return
the expected output types/shapes. They do not validate mathematical
correctness of the homomorphic computation (that would require a full
model comparison against a plaintext reference, which belongs in
integration tests).

All tests are marked `slow` because they involve multiple ciphertext
operations. Run them with:  pytest -m slow
"""

import numpy as np
import pytest

pytest.importorskip("pyOpenFHE")

from conftest import CKKS_ATOL

pytestmark = pytest.mark.slow

# Matrix side length used throughout. Must be a power of 2 for SHIELD sharding.
MTX_SIZE = 4
N_SHARDS = MTX_SIZE  # one shard per row of the matrix


@pytest.fixture(scope="module")
def cnn_cc_keys():
    """Module-scoped CKKS context tuned for CNN operations."""
    from pyOpenFHE import CKKS
    from pyOpenFHE import enums as pal

    batch_size = MTX_SIZE * MTX_SIZE  # 16

    cc = CKKS.genCryptoContextCKKS(
        multiplicativeDepth=10,
        scalingFactorBits=59,
        batchSize=batch_size,
    )
    for feat in (
        pal.PKESchemeFeature.PKE,
        pal.PKESchemeFeature.KEYSWITCH,
        pal.PKESchemeFeature.LEVELEDSHE,
        pal.PKESchemeFeature.ADVANCEDSHE,
    ):
        cc.enable(feat)

    keys = cc.keyGen()
    cc.evalMultKeyGen(keys.secretKey)
    cc.evalPowerOf2RotationKeyGen(keys.secretKey)

    return cc, keys


@pytest.fixture(scope="module")
def shards(cnn_cc_keys):
    """Encrypt N_SHARDS random vectors to use as input shards."""
    cc, keys = cnn_cc_keys
    rng = np.random.default_rng(0)
    batch_size = cc.getBatchSize()
    return [
        cc.encrypt(keys.publicKey, rng.random(batch_size).astype(np.float64))
        for _ in range(N_SHARDS)
    ]


# ── Import smoke test ──────────────────────────────────────────────────────────

class TestCNNImport:
    def test_module_importable(self):
        from pyOpenFHE.CKKS import CNN  # noqa: F401

    def test_omp_set_num_threads(self):
        from pyOpenFHE.CKKS import CNN
        CNN.omp_set_num_threads(1)

    def test_omp_set_nested(self):
        from pyOpenFHE.CKKS import CNN
        CNN.omp_set_nested(False)

    def test_omp_set_dynamic(self):
        from pyOpenFHE.CKKS import CNN
        CNN.omp_set_dynamic(False)


# ── conv2d ─────────────────────────────────────────────────────────────────────

class TestConv2d:
    def test_returns_list(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        from pyOpenFHE.CKKS import CKKSCiphertext

        rng = np.random.default_rng(1)
        filters = rng.random((MTX_SIZE, MTX_SIZE)).astype(np.float64)
        perm = np.eye(MTX_SIZE, dtype=np.float64)

        result = CNN.conv2d(shards, filters, MTX_SIZE, perm)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(ct, CKKSCiphertext) for ct in result)

    def test_output_decryptable(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN

        cc, keys = cnn_cc_keys
        rng = np.random.default_rng(2)
        filters = rng.random((MTX_SIZE, MTX_SIZE)).astype(np.float64)
        perm = np.eye(MTX_SIZE, dtype=np.float64)

        result = CNN.conv2d(shards, filters, MTX_SIZE, perm)
        for ct in result:
            y = cc.decrypt(keys.secretKey, ct)
            assert y is not None


# ── pool ───────────────────────────────────────────────────────────────────────

class TestPool:
    def test_returns_list(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        from pyOpenFHE.CKKS import CKKSCiphertext

        result = CNN.pool(shards, MTX_SIZE, conv=True)
        assert isinstance(result, list)
        assert all(isinstance(ct, CKKSCiphertext) for ct in result)

    def test_output_decryptable(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN

        cc, keys = cnn_cc_keys
        result = CNN.pool(shards, MTX_SIZE, conv=True)
        for ct in result:
            y = cc.decrypt(keys.secretKey, ct)
            assert y is not None


# ── linear ─────────────────────────────────────────────────────────────────────

class TestLinear:
    def test_returns_ciphertext(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        from pyOpenFHE.CKKS import CKKSCiphertext

        rng = np.random.default_rng(3)
        weights = rng.random((MTX_SIZE, MTX_SIZE)).astype(np.float64)
        perm = np.eye(MTX_SIZE, dtype=np.float64)

        result = CNN.linear(shards, weights, MTX_SIZE, perm, pool_factor=1)
        assert isinstance(result, CKKSCiphertext)

    def test_output_decryptable(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN

        cc, keys = cnn_cc_keys
        rng = np.random.default_rng(4)
        weights = rng.random((MTX_SIZE, MTX_SIZE)).astype(np.float64)
        perm = np.eye(MTX_SIZE, dtype=np.float64)

        result = CNN.linear(shards, weights, MTX_SIZE, perm, pool_factor=1)
        y = cc.decrypt(keys.secretKey, result)
        assert y is not None


# ── upsample ───────────────────────────────────────────────────────────────────

class TestUpsample:
    def test_returns_list(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        from pyOpenFHE.CKKS import CKKSCiphertext

        perm = np.eye(MTX_SIZE, dtype=np.float64)
        result = CNN.upsample(shards, MTX_SIZE, perm, upsample_type=0)
        assert isinstance(result, list)
        assert all(isinstance(ct, CKKSCiphertext) for ct in result)


# ── fhe_gelu ───────────────────────────────────────────────────────────────────

class TestFheGelu:
    def test_returns_list(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        from pyOpenFHE.CKKS import CKKSCiphertext

        result = CNN.fhe_gelu(shards, degree=3, bound=5.0)
        assert isinstance(result, list)
        assert all(isinstance(ct, CKKSCiphertext) for ct in result)

    def test_output_decryptable(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN

        cc, keys = cnn_cc_keys
        result = CNN.fhe_gelu(shards, degree=3, bound=5.0)
        for ct in result:
            y = cc.decrypt(keys.secretKey, ct)
            assert y is not None
