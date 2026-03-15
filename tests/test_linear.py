"""Tests for CNN.linear.

linear(shards, weights, mtx_size, permutation, pool_factor)
Weight tensor shape: (num_outputs, num_inputs).
With a single shard and identity permutation, output position r contains
dot(weights[r], plaintext).
"""

import numpy as np
import pytest

pytest.importorskip("pyOpenFHE")

from conftest import CKKS_ATOL, identity_perm

pytestmark = pytest.mark.slow

MTX_SIZE = 4
BATCH_SIZE = MTX_SIZE * MTX_SIZE


class TestLinear:
    # -- smoke --

    def test_returns_ciphertext(self, cnn_cc_keys, cnn_single_shard_pair):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        cc, keys = cnn_cc_keys
        ct, _ = cnn_single_shard_pair
        weights = np.random.default_rng(3).random((1, BATCH_SIZE)).astype(np.float64)
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        assert isinstance(result, CKKSCiphertext)

    def test_output_decryptable(self, cnn_cc_keys, cnn_single_shard_pair):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = cnn_single_shard_pair
        weights = np.random.default_rng(4).random((1, BATCH_SIZE)).astype(np.float64)
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        assert cc.decrypt(keys.secretKey, result) is not None

    # -- correctness --

    def test_zero_weights_gives_zero(self, cnn_cc_keys, cnn_single_shard_pair):
        """A zero weight matrix should produce output ≈ 0."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = cnn_single_shard_pair
        weights = np.zeros((1, BATCH_SIZE))
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        decrypted = np.array(cc.decrypt(keys.secretKey, result))
        assert abs(decrypted[0]) < CKKS_ATOL

    def test_dot_product_correctness(self, cnn_cc_keys, cnn_single_shard_pair):
        """Single-shard linear: output position r = dot(weights[r], plaintext)."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, plaintext = cnn_single_shard_pair
        weights = np.random.default_rng(5).random((1, BATCH_SIZE)).astype(np.float64)
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        decrypted = np.array(cc.decrypt(keys.secretKey, result))
        expected = np.dot(weights[0], plaintext)
        np.testing.assert_allclose(decrypted[0], expected, atol=CKKS_ATOL)

    def test_two_outputs_dot_product_correctness(self, cnn_cc_keys, cnn_single_shard_pair):
        """Multiple output rows are each independently computed dot products."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, plaintext = cnn_single_shard_pair
        rng = np.random.default_rng(6)
        weights = rng.random((2, BATCH_SIZE)).astype(np.float64)
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        decrypted = np.array(cc.decrypt(keys.secretKey, result))
        for r in range(2):
            np.testing.assert_allclose(
                decrypted[r], np.dot(weights[r], plaintext), atol=CKKS_ATOL
            )
