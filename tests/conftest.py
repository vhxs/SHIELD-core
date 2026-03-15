"""
Shared pytest fixtures for pyOpenFHE tests.

All crypto-context and key fixtures are session-scoped because key generation
is expensive. Tests should treat these objects as read-only; each test that
needs a fresh ciphertext should create it locally.
"""

import pytest
import numpy as np


# ── CNN shared helpers ─────────────────────────────────────────────────────────

def identity_perm(n: int) -> np.ndarray:
    """1-D identity channel permutation of length n."""
    return np.arange(n, dtype=np.float64)


# ── CNN shared fixtures ────────────────────────────────────────────────────────
#
# Session-scoped so key generation runs once across all CNN test files.
# The C++ CNN functions copy their shard arguments, so sharing ciphertexts
# between files is safe.

_CNN_BATCH_SIZE = 16  # MTX_SIZE=4, so 4×4 = 16 slots per shard


@pytest.fixture(scope="session")
def cnn_cc_keys():
    """CKKS context tuned for CNN operations (multiplicativeDepth=10)."""
    from pyOpenFHE import CKKS, enums as pal
    cc = CKKS.genCryptoContextCKKS(
        multiplicativeDepth=10,
        scalingFactorBits=59,
        batchSize=_CNN_BATCH_SIZE,
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


@pytest.fixture(scope="session")
def cnn_shards(cnn_cc_keys):
    """4 encrypted random shards (one per input channel) for a 4×4 feature map."""
    cc, keys = cnn_cc_keys
    rng = np.random.default_rng(0)
    return [
        cc.encrypt(keys.publicKey, rng.random(_CNN_BATCH_SIZE).astype(np.float64))
        for _ in range(4)
    ]


@pytest.fixture(scope="session")
def cnn_single_shard_pair(cnn_cc_keys):
    """A single (ciphertext, plaintext) pair with a known random plaintext."""
    cc, keys = cnn_cc_keys
    rng = np.random.default_rng(42)
    plaintext = rng.random(_CNN_BATCH_SIZE).astype(np.float64)
    ct = cc.encrypt(keys.publicKey, plaintext)
    return ct, plaintext


# ── Parameter constants ────────────────────────────────────────────────────────

# CKKS: floating-point approximate HE
CKKS_MULT_DEPTH = 5
CKKS_SCALE_BITS = 59
CKKS_BATCH_SIZE = 16
# Absolute tolerance for approximate CKKS equality checks.
# After a single encrypt/decrypt the error is ~1e-14; after arithmetic it grows.
CKKS_ATOL = 1e-4



# ── CKKS fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def ckks_cc_keys():
    """Session-scoped CKKS CryptoContext with mult + rotation keys."""
    from pyOpenFHE import CKKS
    from pyOpenFHE import enums as pal

    cc = CKKS.genCryptoContextCKKS(
        CKKS_MULT_DEPTH, CKKS_SCALE_BITS, CKKS_BATCH_SIZE
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


@pytest.fixture(scope="session")
def ckks_bootstrap_cc_keys():
    """Session-scoped CKKS context configured for bootstrapping (slow setup)."""
    from pyOpenFHE import CKKS
    from pyOpenFHE import enums as pal

    # Small ring dimension for test speed; security is disabled.
    cc = CKKS.genCryptoContextCKKS(
        26, 59, 64,
        stdLevel=pal.SecurityLevel.HEStd_NotSet,
        ringDim=128,
    )
    for feat in (
        pal.PKESchemeFeature.PKE,
        pal.PKESchemeFeature.KEYSWITCH,
        pal.PKESchemeFeature.LEVELEDSHE,
        pal.PKESchemeFeature.ADVANCEDSHE,
        pal.PKESchemeFeature.FHE,
    ):
        cc.enable(feat)

    keys = cc.keyGen()
    cc.evalMultKeyGen(keys.secretKey)
    cc.evalPowerOf2RotationKeyGen(keys.secretKey)
    cc.evalBootstrapSetup()
    cc.evalBootstrapKeyGen(keys.secretKey)

    return cc, keys
