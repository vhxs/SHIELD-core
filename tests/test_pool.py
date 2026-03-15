"""Tests for CNN.pool.

pool(shards, mtx_size, True)  → average pooling (2×2 window)
pool(shards, mtx_size, False) → spatial downsampling only (no averaging)
With MTX_SIZE=4 and N_CHANNELS=4 input shards, the output is 1 shard.
"""

import numpy as np
import pytest

pytest.importorskip("pyOpenFHE")

from conftest import CKKS_ATOL

pytestmark = pytest.mark.slow

MTX_SIZE = 4
BATCH_SIZE = MTX_SIZE * MTX_SIZE
N_CHANNELS = MTX_SIZE


class TestPool:
    # -- smoke --

    def test_returns_list(self, cnn_cc_keys, cnn_shards):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        result = CNN.pool(cnn_shards, MTX_SIZE, True)
        assert isinstance(result, list)
        assert all(isinstance(c, CKKSCiphertext) for c in result)

    def test_output_decryptable_conv_true(self, cnn_cc_keys, cnn_shards):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.pool(cnn_shards, MTX_SIZE, True)
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    def test_output_decryptable_conv_false(self, cnn_cc_keys, cnn_shards):
        """pool with False (downsample only) should also produce valid output."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.pool(cnn_shards, MTX_SIZE, False)
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    # -- correctness --

    def test_output_shard_count_reduces(self, cnn_cc_keys, cnn_shards):
        """4 input shards with MTX_SIZE=4 should consolidate to 1 output shard."""
        from pyOpenFHE.CKKS import CNN
        result = CNN.pool(cnn_shards, MTX_SIZE, True)
        assert len(result) == 1

    def test_conv_true_and_false_differ(self, cnn_cc_keys, cnn_shards):
        """Average pooling (True) and downsample-only (False) differ numerically."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        y_avg = np.array(cc.decrypt(keys.secretKey, CNN.pool(cnn_shards, MTX_SIZE, True)[0]))
        y_ds  = np.array(cc.decrypt(keys.secretKey, CNN.pool(cnn_shards, MTX_SIZE, False)[0]))
        assert not np.allclose(y_avg, y_ds)

    def test_uniform_input_average_pool_preserves_value(self, cnn_cc_keys):
        """Average pool of a spatially uniform image should output the same constant.

        For uniform input c: the 2×2 pre-convolution sums to 4c (cyclic), the
        horizontal/vertical reduce × 0.25 restores c, and consolidation fills the
        output shard uniformly with c.
        """
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        c = 0.75
        uniform_shards = [
            cc.encrypt(keys.publicKey, np.full(BATCH_SIZE, c))
            for _ in range(N_CHANNELS)
        ]
        result = CNN.pool(uniform_shards, MTX_SIZE, True)
        decrypted = np.array(cc.decrypt(keys.secretKey, result[0]))
        np.testing.assert_allclose(decrypted, c, atol=CKKS_ATOL)
