"""
Shared pytest fixtures for pyOpenFHE tests.

All crypto-context and key fixtures are session-scoped because key generation
is expensive. Tests should treat these objects as read-only; each test that
needs a fresh ciphertext should create it locally.
"""

import pytest
import numpy as np


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
