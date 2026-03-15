"""
Tests for CKKS CNN operations.

Organised in two tiers:
  - Smoke tests: verify each function is callable and returns the right type.
  - Correctness tests: encrypt a known plaintext, run the operation, decrypt,
    and compare against an independent plaintext reference implementation.

All tests are marked `slow` because they involve multiple ciphertext operations.
Run with:  pytest -m slow
"""

import math
import numpy as np
import pytest

pytest.importorskip("pyOpenFHE")

from conftest import CKKS_ATOL

pytestmark = pytest.mark.slow

MTX_SIZE = 4          # spatial dimension of the (square) feature map
BATCH_SIZE = MTX_SIZE * MTX_SIZE   # 16 slots per shard
N_CHANNELS = MTX_SIZE              # 4 input channels


# ── Plaintext reference helpers ────────────────────────────────────────────────

def _gelu(x: float) -> float:
    """Exact GELU: x * Φ(x) where Φ is the standard normal CDF."""
    return x * 0.5 * math.erfc(-x * math.sqrt(0.5))


def _gelu_scaled(x: float, bound: float) -> float:
    """fhe_gelu approximates gelu(x * bound) on the domain [-1, 1]."""
    return _gelu(x * bound)


def identity_perm(n: int) -> np.ndarray:
    """1-D identity channel permutation of length n."""
    return np.arange(n, dtype=np.float64)


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


# ── Module-scoped fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cnn_cc_keys():
    """CKKS context tuned for CNN operations (multiplicativeDepth=10)."""
    from pyOpenFHE import CKKS, enums as pal
    cc = CKKS.genCryptoContextCKKS(
        multiplicativeDepth=10,
        scalingFactorBits=59,
        batchSize=BATCH_SIZE,
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
    """N_CHANNELS encrypted random shards (one per input channel)."""
    cc, keys = cnn_cc_keys
    rng = np.random.default_rng(0)
    return [
        cc.encrypt(keys.publicKey, rng.random(BATCH_SIZE).astype(np.float64))
        for _ in range(N_CHANNELS)
    ]


@pytest.fixture(scope="module")
def single_shard_pair(cnn_cc_keys):
    """A single (ciphertext, plaintext) pair with a known random plaintext."""
    cc, keys = cnn_cc_keys
    rng = np.random.default_rng(42)
    plaintext = rng.random(BATCH_SIZE).astype(np.float64)
    ct = cc.encrypt(keys.publicKey, plaintext)
    return ct, plaintext


# ── Import smoke tests ─────────────────────────────────────────────────────────

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
#
# Filter tensor shape: (in_channels, out_channels, ker_h, ker_w).
# Permutation: 1-D int vector of length in_channels mapping physical channel
# index → logical channel index into the filter tensor.

class TestConv2d:
    # -- smoke --

    def test_returns_list(self, cnn_cc_keys, single_shard_pair):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        cc, keys = cnn_cc_keys
        ct, _ = single_shard_pair
        filters = np.random.default_rng(1).random((1, 1, 1, 1)).astype(np.float64)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(c, CKKSCiphertext) for c in result)

    def test_output_decryptable(self, cnn_cc_keys, single_shard_pair):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = single_shard_pair
        filters = np.random.default_rng(2).random((1, 1, 1, 1)).astype(np.float64)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    # -- correctness --

    def test_1x1_kernel_scales_input(self, cnn_cc_keys, single_shard_pair):
        """1×1 conv with weight w should multiply every pixel by w."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, plaintext = single_shard_pair
        w = 3.0
        filters = np.array([[[[w]]]]).astype(np.float64)  # (1, 1, 1, 1)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        decrypted = np.array(cc.decrypt(keys.secretKey, result[0]))
        np.testing.assert_allclose(
            decrypted[:BATCH_SIZE], w * plaintext, atol=CKKS_ATOL
        )

    def test_zero_filter_gives_zero_output(self, cnn_cc_keys, single_shard_pair):
        """A filter of all zeros should produce an all-zero output."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = single_shard_pair
        filters = np.zeros((1, 1, 1, 1))
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        decrypted = np.array(cc.decrypt(keys.secretKey, result[0]))
        np.testing.assert_allclose(decrypted[:BATCH_SIZE], 0.0, atol=CKKS_ATOL)

    def test_two_output_channels_returns_two_shards(self, cnn_cc_keys, single_shard_pair):
        """With 2 output channels and 1 physical channel per shard, expect 2 output shards."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = single_shard_pair
        filters = np.random.default_rng(3).random((1, 2, 1, 1)).astype(np.float64)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        assert len(result) == 2

    def test_two_output_channels_are_independent(self, cnn_cc_keys, single_shard_pair):
        """Each output channel is a separate scaled copy — they differ when weights differ."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = single_shard_pair
        filters = np.array([[[[2.0]], [[5.0]]]]).astype(np.float64)  # (1, 2, 1, 1)
        result = CNN.conv2d([ct], filters, MTX_SIZE, identity_perm(1))
        y0 = np.array(cc.decrypt(keys.secretKey, result[0]))
        y1 = np.array(cc.decrypt(keys.secretKey, result[1]))
        assert not np.allclose(y0[:BATCH_SIZE], y1[:BATCH_SIZE], atol=CKKS_ATOL)

    def test_3x3_kernel_spatial_correctness(self, cnn_cc_keys, single_shard_pair):
        """3×3 cross-correlation with zero-padding matches the plaintext reference.

        This exercises the actual sliding-window convolution: each output pixel
        accumulates a weighted sum of its 3×3 neighbourhood, with zero-padding
        at the image boundary.  A 1×1 kernel test cannot catch mistakes in the
        rotation offsets or boundary masking.
        """
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, plaintext = single_shard_pair

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


# ── pool ───────────────────────────────────────────────────────────────────────
#
# pool(shards, mtx_size, True)  → average pooling (2×2 window)
# pool(shards, mtx_size, False) → spatial downsampling only (no averaging)
# With MTX_SIZE=4 and N_CHANNELS=4 input shards, the output is 1 shard.

class TestPool:
    # -- smoke --

    def test_returns_list(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        result = CNN.pool(shards, MTX_SIZE, True)
        assert isinstance(result, list)
        assert all(isinstance(c, CKKSCiphertext) for c in result)

    def test_output_decryptable_conv_true(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.pool(shards, MTX_SIZE, True)
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    def test_output_decryptable_conv_false(self, cnn_cc_keys, shards):
        """pool with False (downsample only) should also produce valid output."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.pool(shards, MTX_SIZE, False)
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    # -- correctness --

    def test_output_shard_count_reduces(self, cnn_cc_keys, shards):
        """4 input shards with MTX_SIZE=4 should consolidate to 1 output shard."""
        from pyOpenFHE.CKKS import CNN
        result = CNN.pool(shards, MTX_SIZE, True)
        assert len(result) == 1

    def test_conv_true_and_false_differ(self, cnn_cc_keys, shards):
        """Average pooling (True) and downsample-only (False) differ numerically."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        y_avg = np.array(cc.decrypt(keys.secretKey, CNN.pool(shards, MTX_SIZE, True)[0]))
        y_ds  = np.array(cc.decrypt(keys.secretKey, CNN.pool(shards, MTX_SIZE, False)[0]))
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


# ── linear ─────────────────────────────────────────────────────────────────────
#
# linear(shards, weights, mtx_size, permutation, pool_factor)
# Weight tensor shape: (num_outputs, num_inputs).
# With a single shard and identity permutation, output position r contains
# dot(weights[r], plaintext).

class TestLinear:
    # -- smoke --

    def test_returns_ciphertext(self, cnn_cc_keys, single_shard_pair):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        cc, keys = cnn_cc_keys
        ct, _ = single_shard_pair
        weights = np.random.default_rng(3).random((1, BATCH_SIZE)).astype(np.float64)
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        assert isinstance(result, CKKSCiphertext)

    def test_output_decryptable(self, cnn_cc_keys, single_shard_pair):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = single_shard_pair
        weights = np.random.default_rng(4).random((1, BATCH_SIZE)).astype(np.float64)
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        assert cc.decrypt(keys.secretKey, result) is not None

    # -- correctness --

    def test_zero_weights_gives_zero(self, cnn_cc_keys, single_shard_pair):
        """A zero weight matrix should produce output ≈ 0."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, _ = single_shard_pair
        weights = np.zeros((1, BATCH_SIZE))
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        decrypted = np.array(cc.decrypt(keys.secretKey, result))
        assert abs(decrypted[0]) < CKKS_ATOL

    def test_dot_product_correctness(self, cnn_cc_keys, single_shard_pair):
        """Single-shard linear: output position r = dot(weights[r], plaintext)."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, plaintext = single_shard_pair
        weights = np.random.default_rng(5).random((1, BATCH_SIZE)).astype(np.float64)
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        decrypted = np.array(cc.decrypt(keys.secretKey, result))
        expected = np.dot(weights[0], plaintext)
        np.testing.assert_allclose(decrypted[0], expected, atol=CKKS_ATOL)

    def test_two_outputs_dot_product_correctness(self, cnn_cc_keys, single_shard_pair):
        """Multiple output rows are each independently computed dot products."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        ct, plaintext = single_shard_pair
        rng = np.random.default_rng(6)
        weights = rng.random((2, BATCH_SIZE)).astype(np.float64)
        result = CNN.linear([ct], weights, MTX_SIZE, identity_perm(1), 1)
        decrypted = np.array(cc.decrypt(keys.secretKey, result))
        for r in range(2):
            np.testing.assert_allclose(
                decrypted[r], np.dot(weights[r], plaintext), atol=CKKS_ATOL
            )


# ── upsample ───────────────────────────────────────────────────────────────────
#
# upsample_type=0: bed-of-nails (insert zeros, 2× spatial expansion each axis)
# upsample_type=1: nearest-neighbor (replicate values into the inserted positions)
# Any other upsample_type raises RuntimeError.

class TestUpsample:
    # -- smoke --

    def test_returns_list(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        result = CNN.upsample(shards, MTX_SIZE, identity_perm(N_CHANNELS), 0)
        assert isinstance(result, list)
        assert all(isinstance(c, CKKSCiphertext) for c in result)

    def test_output_decryptable_bed_of_nails(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.upsample(shards, MTX_SIZE, identity_perm(N_CHANNELS), 0)
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    def test_output_decryptable_nearest_neighbor(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.upsample(shards, MTX_SIZE, identity_perm(N_CHANNELS), 1)
        for c in result:
            assert cc.decrypt(keys.secretKey, c) is not None

    # -- correctness --

    def test_output_shard_count_increases(self, cnn_cc_keys, shards):
        """Upsampling should produce more output shards than input shards."""
        from pyOpenFHE.CKKS import CNN
        result = CNN.upsample(shards, MTX_SIZE, identity_perm(N_CHANNELS), 0)
        assert len(result) > len(shards)

    def test_invalid_upsample_type_raises(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        with pytest.raises(RuntimeError):
            CNN.upsample(shards, MTX_SIZE, identity_perm(N_CHANNELS), 2)

    def test_bed_of_nails_and_nearest_neighbor_differ(self, cnn_cc_keys, shards):
        """Bed-of-nails (zeros) and nearest-neighbor (replicated) produce different values."""
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        bon = CNN.upsample(shards, MTX_SIZE, identity_perm(N_CHANNELS), 0)
        nn  = CNN.upsample(shards, MTX_SIZE, identity_perm(N_CHANNELS), 1)
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


# ── fhe_gelu ───────────────────────────────────────────────────────────────────
#
# fhe_gelu evaluates a Chebyshev approximation of gelu(x * bound) on [-1, 1].
# Raises RuntimeError if getTowersRemaining() is too low for the requested degree.

class TestFheGelu:
    # -- smoke --

    def test_returns_list(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN, CKKSCiphertext
        result = CNN.fhe_gelu(shards, 3, 5.0)
        assert isinstance(result, list)
        assert all(isinstance(c, CKKSCiphertext) for c in result)

    def test_output_decryptable(self, cnn_cc_keys, shards):
        from pyOpenFHE.CKKS import CNN
        cc, keys = cnn_cc_keys
        result = CNN.fhe_gelu(shards, 3, 5.0)
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
