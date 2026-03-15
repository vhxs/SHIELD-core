"""Tests for CNN.upsample.

upsample_type=0: bed-of-nails (insert zeros, 2× spatial expansion each axis)
upsample_type=1: nearest-neighbor (replicate values into the inserted positions)
Any other upsample_type raises RuntimeError.
"""

import numpy as np
import pytest

pytest.importorskip("pyOpenFHE")

from conftest import CKKS_ATOL, identity_perm

pytestmark = pytest.mark.slow

MTX_SIZE = 4
BATCH_SIZE = MTX_SIZE * MTX_SIZE
N_CHANNELS = MTX_SIZE


class TestUpsample:
    # -- smoke --

    def test_returns_list(self, cnn_cc_keys, cnn_shards):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        result = CNN.upsample(cnn_shards, MTX_SIZE, identity_perm(N_CHANNELS), 0)
        assert isinstance(result, list)
        assert all(isinstance(c, CKKSCiphertext) for c in result)

    def test_output_decryptable_bed_of_nails(self, cnn_cc_keys, cnn_shards):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.upsample(cnn_shards, MTX_SIZE, identity_perm(N_CHANNELS), 0)
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    def test_output_decryptable_nearest_neighbor(self, cnn_cc_keys, cnn_shards):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.upsample(cnn_shards, MTX_SIZE, identity_perm(N_CHANNELS), 1)
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    # -- correctness --

    def test_output_shard_count_increases(self, cnn_cc_keys, cnn_shards):
        """Upsampling should produce more output shards than input shards."""
        from pyOpenFHE.CKKS import CNN
        result = CNN.upsample(cnn_shards, MTX_SIZE, identity_perm(N_CHANNELS), 0)
        assert len(result) > len(cnn_shards)

    def test_invalid_upsample_type_raises(self, cnn_cc_keys, cnn_shards):
        from pyOpenFHE.CKKS import CNN
        with pytest.raises(RuntimeError):
            CNN.upsample(cnn_shards, MTX_SIZE, identity_perm(N_CHANNELS), 2)

    def test_bed_of_nails_and_nearest_neighbor_differ(self, cnn_cc_keys, cnn_shards):
        """Bed-of-nails (zeros) and nearest-neighbor (replicated) produce different values."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        bon = CNN.upsample(cnn_shards, MTX_SIZE, identity_perm(N_CHANNELS), 0)
        nn  = CNN.upsample(cnn_shards, MTX_SIZE, identity_perm(N_CHANNELS), 1)
        y_bon = np.array(cc.decrypt(keys.secretKey, bon[0]))
        y_nn  = np.array(cc.decrypt(keys.secretKey, nn[0]))
        assert not np.allclose(y_bon, y_nn)

    def test_uniform_input_nearest_neighbor_preserves_value(self, cnn_cc_keys):
        """Nearest-neighbor upsample of a uniform image should output the same constant.

        Bed-of-nails places each pixel at even (row, col) slots in the 2× output;
        nearest-neighbor then fills the interleaved slots via two right-shifts, so
        every slot in every output shard ends up equal to the original constant c.
        """
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        c = 0.6
        uniform_shards = [
            cc.encrypt(keys.publicKey, np.full(BATCH_SIZE, c))
            for _ in range(N_CHANNELS)
        ]
        result = CNN.upsample(uniform_shards, MTX_SIZE, identity_perm(N_CHANNELS), 1)
        for out_shard in result:
            decrypted = np.array(cc.decrypt(keys.secretKey, out_shard))
            np.testing.assert_allclose(decrypted, c, atol=CKKS_ATOL)

    # TODO: add a value-level correctness test for bed-of-nails (upsample_type=0).
    # This requires knowing the exact output slot layout after upsample_vertical_expand
    # and upsample_horizontal_expand (4 input shards × 4 sub-shards = 16 output shards,
    # each holding 1 expanded row of the 2× image). Deferred until upsample is in active
    # use in the downstream CNN pipeline.
