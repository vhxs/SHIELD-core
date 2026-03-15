"""Tests for the CKKS homomorphic encryption scheme (floating-point)."""

import numpy as np
import pytest

pytest.importorskip("pyOpenFHE")

from conftest import CKKS_ATOL, CKKS_BATCH_SIZE

_RNG = np.random.default_rng(42)


def _rand(n=CKKS_BATCH_SIZE) -> np.ndarray:
    """Reproducible random floats in [0, 1)."""
    return _RNG.random(n).astype(np.float64)


def _enc(cc, keys, x):
    return cc.encrypt(keys.publicKey, x)


def _dec(cc, keys, c) -> np.ndarray:
    """Decrypt and return only the active slots."""
    return cc.decrypt(keys.secretKey, c)[:CKKS_BATCH_SIZE]


# ── Context properties ─────────────────────────────────────────────────────────

class TestCKKSContext:
    def test_batch_size(self, ckks_cc_keys):
        cc, _ = ckks_cc_keys
        assert cc.getBatchSize() == CKKS_BATCH_SIZE

    def test_ring_dimension_positive(self, ckks_cc_keys):
        cc, _ = ckks_cc_keys
        assert cc.getRingDimension() > 0

    def test_scheme_id(self, ckks_cc_keys):
        from pyOpenFHE import enums as pal
        cc, _ = ckks_cc_keys
        assert cc.getSchemeID() == pal.SCHEME.CKKSRNS_SCHEME

    def test_keypair_valid(self, ckks_cc_keys):
        _, keys = ckks_cc_keys
        assert keys.good()


# ── Encrypt / decrypt round-trips ─────────────────────────────────────────────

class TestCKKSEncryptDecrypt:
    def test_roundtrip_ndarray(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        np.testing.assert_allclose(_dec(cc, keys, _enc(cc, keys, x)), x, atol=CKKS_ATOL)

    def test_roundtrip_list(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand().tolist()
        y = _dec(cc, keys, _enc(cc, keys, x))
        np.testing.assert_allclose(y, x, atol=CKKS_ATOL)

    def test_encrypt_with_secret_key(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = cc.encrypt(keys.secretKey, x)
        np.testing.assert_allclose(_dec(cc, keys, c), x, atol=CKKS_ATOL)

    def test_zero_vector(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = np.zeros(CKKS_BATCH_SIZE)
        np.testing.assert_allclose(_dec(cc, keys, _enc(cc, keys, x)), x, atol=CKKS_ATOL)

    def test_ones_vector(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = np.ones(CKKS_BATCH_SIZE)
        np.testing.assert_allclose(_dec(cc, keys, _enc(cc, keys, x)), x, atol=CKKS_ATOL)

    def test_negative_values(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand() - 0.5  # values in [-0.5, 0.5)
        np.testing.assert_allclose(_dec(cc, keys, _enc(cc, keys, x)), x, atol=CKKS_ATOL)


# ── Ciphertext properties ──────────────────────────────────────────────────────

class TestCKKSCiphertextProperties:
    def test_batch_size(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        c = _enc(cc, keys, _rand())
        assert c.getBatchSize() == CKKS_BATCH_SIZE

    def test_mult_level_fresh(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        c = _enc(cc, keys, _rand())
        assert c.getMultLevel() == 0

    def test_mult_level_increases_after_multiply(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x) * _enc(cc, keys, x)
        assert c.getMultLevel() > 0

    def test_towers_remaining_positive(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        c = _enc(cc, keys, _rand())
        assert c.getTowersRemaining() > 0

    def test_get_crypto_context(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        c = _enc(cc, keys, _rand())
        # Should not raise; returns the context that created this ciphertext
        ctx = c.getCryptoContext()
        assert ctx is not None

    def test_copy_constructor(self, ckks_cc_keys):
        from pyOpenFHE.CKKS import CKKSCiphertext
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x)
        c_copy = CKKSCiphertext(c)
        np.testing.assert_allclose(_dec(cc, keys, c_copy), x, atol=CKKS_ATOL)

    def test_equality_self(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        c = _enc(cc, keys, _rand())
        assert c == c

    def test_inequality_distinct(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        c1 = _enc(cc, keys, _rand())
        c2 = _enc(cc, keys, _rand())
        assert c1 != c2


# ── Ciphertext–ciphertext arithmetic ──────────────────────────────────────────

class TestCKKSCiphertextArithmetic:
    def test_add(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1) + _enc(cc, keys, x2)
        np.testing.assert_allclose(_dec(cc, keys, c), x1 + x2, atol=CKKS_ATOL)

    def test_sub(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1) - _enc(cc, keys, x2)
        np.testing.assert_allclose(_dec(cc, keys, c), x1 - x2, atol=CKKS_ATOL)

    def test_mul(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1) * _enc(cc, keys, x2)
        np.testing.assert_allclose(_dec(cc, keys, c), x1 * x2, atol=CKKS_ATOL)

    def test_add_inplace(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1)
        c += _enc(cc, keys, x2)
        np.testing.assert_allclose(_dec(cc, keys, c), x1 + x2, atol=CKKS_ATOL)

    def test_sub_inplace(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1)
        c -= _enc(cc, keys, x2)
        np.testing.assert_allclose(_dec(cc, keys, c), x1 - x2, atol=CKKS_ATOL)

    def test_mul_inplace(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1)
        c *= _enc(cc, keys, x2)
        np.testing.assert_allclose(_dec(cc, keys, c), x1 * x2, atol=CKKS_ATOL)

    def test_negate(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = -_enc(cc, keys, x)
        np.testing.assert_allclose(_dec(cc, keys, c), -x, atol=CKKS_ATOL)

    def test_add_sub_identity(self, ckks_cc_keys):
        """(c1 + c2) - c2 ≈ c1."""
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1) + _enc(cc, keys, x2) - _enc(cc, keys, x2)
        np.testing.assert_allclose(_dec(cc, keys, c), x1, atol=CKKS_ATOL)


# ── Ciphertext–scalar arithmetic ───────────────────────────────────────────────

class TestCKKSScalarArithmetic:
    def test_add_scalar_right(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x) + 1.5
        np.testing.assert_allclose(_dec(cc, keys, c), x + 1.5, atol=CKKS_ATOL)

    def test_add_scalar_left(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = 1.5 + _enc(cc, keys, x)
        np.testing.assert_allclose(_dec(cc, keys, c), x + 1.5, atol=CKKS_ATOL)

    def test_sub_scalar_right(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x) - 0.5
        np.testing.assert_allclose(_dec(cc, keys, c), x - 0.5, atol=CKKS_ATOL)

    def test_sub_scalar_left(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = 1.0 - _enc(cc, keys, x)
        np.testing.assert_allclose(_dec(cc, keys, c), 1.0 - x, atol=CKKS_ATOL)

    def test_mul_scalar_right(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x) * 3.14
        np.testing.assert_allclose(_dec(cc, keys, c), x * 3.14, atol=CKKS_ATOL)

    def test_mul_scalar_left(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = 2.0 * _enc(cc, keys, x)
        np.testing.assert_allclose(_dec(cc, keys, c), 2.0 * x, atol=CKKS_ATOL)

    def test_add_scalar_inplace(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x)
        c += 0.5
        np.testing.assert_allclose(_dec(cc, keys, c), x + 0.5, atol=CKKS_ATOL)

    def test_sub_scalar_inplace(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x)
        c -= 0.25
        np.testing.assert_allclose(_dec(cc, keys, c), x - 0.25, atol=CKKS_ATOL)

    def test_mul_scalar_inplace(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x)
        c *= 2.0
        np.testing.assert_allclose(_dec(cc, keys, c), x * 2.0, atol=CKKS_ATOL)

    def test_mul_by_zero(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x) * 0.0
        np.testing.assert_allclose(_dec(cc, keys, c), np.zeros(CKKS_BATCH_SIZE), atol=CKKS_ATOL)


# ── Ciphertext–array arithmetic ────────────────────────────────────────────────

class TestCKKSArrayArithmetic:
    def test_add_ndarray_right(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1) + x2
        np.testing.assert_allclose(_dec(cc, keys, c), x1 + x2, atol=CKKS_ATOL)

    def test_add_ndarray_left(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = x1 + _enc(cc, keys, x2)
        np.testing.assert_allclose(_dec(cc, keys, c), x1 + x2, atol=CKKS_ATOL)

    def test_sub_ndarray_right(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1) - x2
        np.testing.assert_allclose(_dec(cc, keys, c), x1 - x2, atol=CKKS_ATOL)

    def test_mul_ndarray_right(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = _enc(cc, keys, x1) * x2
        np.testing.assert_allclose(_dec(cc, keys, c), x1 * x2, atol=CKKS_ATOL)

    def test_mul_ndarray_left(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1, x2 = _rand(), _rand()
        c = x1 * _enc(cc, keys, x2)
        np.testing.assert_allclose(_dec(cc, keys, c), x1 * x2, atol=CKKS_ATOL)

    def test_add_list_right(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1 = _rand()
        x2 = _rand().tolist()
        c = _enc(cc, keys, x1) + x2
        np.testing.assert_allclose(_dec(cc, keys, c), x1 + np.array(x2), atol=CKKS_ATOL)

    def test_mul_list_right(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x1 = _rand()
        x2 = _rand().tolist()
        c = _enc(cc, keys, x1) * x2
        np.testing.assert_allclose(_dec(cc, keys, c), x1 * np.array(x2), atol=CKKS_ATOL)


# ── Rotation ───────────────────────────────────────────────────────────────────

class TestCKKSRotation:
    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_rotate_roundtrip(self, ckks_cc_keys, k):
        """Left-rotate by k then right-rotate by k must recover the original."""
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x)
        y = _dec(cc, keys, (c << k) >> k)
        np.testing.assert_allclose(y, x, atol=CKKS_ATOL)

    def test_left_rotate_by_1(self, ckks_cc_keys):
        """Left rotation by 1: slot i receives value from slot i+1 (cyclic)."""
        cc, keys = ckks_cc_keys
        x = _rand()
        y = _dec(cc, keys, _enc(cc, keys, x) << 1)
        np.testing.assert_allclose(y, np.roll(x, -1), atol=CKKS_ATOL)

    def test_right_rotate_by_1(self, ckks_cc_keys):
        """Right rotation by 1: slot i receives value from slot i-1 (cyclic)."""
        cc, keys = ckks_cc_keys
        x = _rand()
        y = _dec(cc, keys, _enc(cc, keys, x) >> 1)
        np.testing.assert_allclose(y, np.roll(x, 1), atol=CKKS_ATOL)

    def test_left_rotate_inplace(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x)
        c <<= 1
        np.testing.assert_allclose(_dec(cc, keys, c), np.roll(x, -1), atol=CKKS_ATOL)

    def test_right_rotate_inplace(self, ckks_cc_keys):
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x)
        c >>= 1
        np.testing.assert_allclose(_dec(cc, keys, c), np.roll(x, 1), atol=CKKS_ATOL)

    def test_double_left_rotate(self, ckks_cc_keys):
        """Rotating left by 2 equals rotating left by 1 twice."""
        cc, keys = ckks_cc_keys
        x = _rand()
        c = _enc(cc, keys, x)
        y_once = _dec(cc, keys, c << 2)
        # Re-encrypt x to get a fresh ciphertext for the two-step rotation
        c2 = _enc(cc, keys, x)
        y_twice = _dec(cc, keys, (c2 << 1) << 1)
        np.testing.assert_allclose(y_once, y_twice, atol=CKKS_ATOL)


# ── Utility functions ──────────────────────────────────────────────────────────

class TestCKKSUtils:
    def test_zero_pad_short_array(self, ckks_cc_keys):
        cc, _ = ckks_cc_keys
        x = np.array([1.0, 2.0, 3.0])
        padded = cc.zeroPadToBatchSize(x)
        assert len(padded) == CKKS_BATCH_SIZE
        np.testing.assert_allclose(padded[:3], x)
        np.testing.assert_allclose(padded[3:], 0.0)

    def test_zero_pad_list(self, ckks_cc_keys):
        cc, _ = ckks_cc_keys
        x = [1.0, 2.0]
        padded = cc.zeroPadToBatchSize(x)
        assert len(padded) == CKKS_BATCH_SIZE

    def test_zero_pad_full_length_unchanged(self, ckks_cc_keys):
        cc, _ = ckks_cc_keys
        x = _rand()
        padded = cc.zeroPadToBatchSize(x)
        np.testing.assert_allclose(padded, x)


# ── Bootstrapping (slow) ───────────────────────────────────────────────────────

@pytest.mark.slow
class TestCKKSBootstrapping:
    def test_bootstrap_preserves_value(self, ckks_bootstrap_cc_keys):
        """Bootstrap should recover the plaintext value (within loose tolerance)."""
        cc, keys = ckks_bootstrap_cc_keys
        batch_size = cc.getBatchSize()
        x = np.random.default_rng(7).random(batch_size)
        c = cc.encrypt(keys.publicKey, x)
        c_boot = cc.evalBootstrap(c)
        y = cc.decrypt(keys.secretKey, c_boot)[:batch_size]
        # Bootstrapping introduces more noise; use a looser tolerance.
        np.testing.assert_allclose(y, x, atol=1e-2)

    def test_bootstrap_restores_towers(self, ckks_bootstrap_cc_keys):
        """After exhausting depth, bootstrapping should increase towers remaining."""
        cc, keys = ckks_bootstrap_cc_keys
        batch_size = cc.getBatchSize()
        x = np.random.default_rng(8).random(batch_size)
        c = cc.encrypt(keys.publicKey, x)

        towers_initial = c.getTowersRemaining()
        # Exhaust most of the depth
        for _ in range(20):
            c = c * 1.0

        towers_depleted = c.getTowersRemaining()
        assert towers_depleted < towers_initial

        c_boot = cc.evalBootstrap(c)
        assert c_boot.getTowersRemaining() > towers_depleted
