"""Tests for CNN.conv2d.

Filter tensor shape: (in_channels, out_channels, ker_h, ker_w).
Permutation: 1-D int vector of length in_channels mapping physical channel
index → logical channel index into the filter tensor.
"""

import numpy as np
import pytest

pytest.importorskip("pyOpenFHE")

from conftest import CKKS_ATOL, identity_perm

pytestmark = pytest.mark.slow

MTX_SIZE = 4
BATCH_SIZE = MTX_SIZE * MTX_SIZE


def _ref_conv2d(image_flat: np.ndarray, kernel: np.ndarray, mtx_size: int) -> np.ndarray:
    """Zero-padded 2D cross-correlation matching the FHE implementation.

    The FHE conv uses cyclic ciphertext rotations masked to zero out positions
    that wrapped around the boundary, which is equivalent to zero-padding.
    """
    image = image_flat.reshape(mtx_size, mtx_size)
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    output = np.zeros_like(image)
    for r in range(mtx_size):
        for c in range(mtx_size):
            for ki in range(kernel.shape[0]):
                for kj in range(kernel.shape[1]):
                    ir, ic = r + ki - pad_h, c + kj - pad_w
                    if 0 <= ir < mtx_size and 0 <= ic < mtx_size:
                        output[r, c] += kernel[ki, kj] * image[ir, ic]
    return output.ravel()


class TestConv2d:
    # -- smoke --

    def test_returns_list(self, cnn_cc_keys, cnn_single_shard_pair):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        cc, keys = cnn_cc_keys
        ct, _ = cnn_single_shard_pair
        filters = np.random.default_rng(1).random((1, 1, 1, 1)).astype(np.float64)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(c, CKKSCiphertext) for c in result)

    def test_output_decryptable(self, cnn_cc_keys, cnn_single_shard_pair):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = cnn_single_shard_pair
        filters = np.random.default_rng(2).random((1, 1, 1, 1)).astype(np.float64)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    # -- correctness --

    def test_1x1_kernel_scales_input(self, cnn_cc_keys, cnn_single_shard_pair):
        """1×1 conv with weight w should multiply every pixel by w."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, plaintext = cnn_single_shard_pair
        w = 3.0
        filters = np.array([[[[w]]]]).astype(np.float64)  # (1, 1, 1, 1)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        decrypted = np.array(cc.decrypt(keys.secretKey, result[0]))
        np.testing.assert_allclose(decrypted[:BATCH_SIZE], w * plaintext, atol=CKKS_ATOL)

    def test_zero_filter_gives_zero_output(self, cnn_cc_keys, cnn_single_shard_pair):
        """A filter of all zeros should produce an all-zero output."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = cnn_single_shard_pair
        filters = np.zeros((1, 1, 1, 1))
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        decrypted = np.array(cc.decrypt(keys.secretKey, result[0]))
        np.testing.assert_allclose(decrypted[:BATCH_SIZE], 0.0, atol=CKKS_ATOL)

    def test_two_output_channels_returns_two_shards(self, cnn_cc_keys, cnn_single_shard_pair):
        """With 2 output channels and 1 physical channel per shard, expect 2 output shards."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = cnn_single_shard_pair
        filters = np.random.default_rng(3).random((1, 2, 1, 1)).astype(np.float64)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        assert len(result) == 2

    def test_two_output_channels_are_independent(self, cnn_cc_keys, cnn_single_shard_pair):
        """Each output channel is a separate scaled copy — they differ when weights differ."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = cnn_single_shard_pair
        filters = np.array([[[[2.0]], [[5.0]]]]).astype(np.float64)  # (1, 2, 1, 1)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        y0 = np.array(cc.decrypt(keys.secretKey, result[0]))
        y1 = np.array(cc.decrypt(keys.secretKey, result[1]))
        assert not np.allclose(y0[:BATCH_SIZE], y1[:BATCH_SIZE], atol=CKKS_ATOL)

    def test_3x3_kernel_spatial_correctness(self, cnn_cc_keys, cnn_single_shard_pair):
        """3×3 cross-correlation with zero-padding matches the plaintext reference.

        This exercises the actual sliding-window convolution: each output pixel
        accumulates a weighted sum of its 3×3 neighbourhood, with zero-padding
        at the image boundary.  A 1×1 kernel test cannot catch mistakes in the
        rotation offsets or boundary masking.
        """
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, plaintext = cnn_single_shard_pair

        kernel = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ], dtype=np.float64)
        filters = kernel[np.newaxis, np.newaxis, :, :]  # shape (1, 1, 3, 3)

        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        decrypted = np.array(cc.decrypt(keys.secretKey, result[0]))[:BATCH_SIZE]

        expected = _ref_conv2d(plaintext, kernel, MTX_SIZE)
        np.testing.assert_allclose(decrypted, expected, atol=CKKS_ATOL)
