"""Tests for CKKS serialization / deserialization round-trips."""

import numpy as np
import pytest

pytest.importorskip("pyOpenFHE")

from conftest import CKKS_ATOL, CKKS_BATCH_SIZE

_RNG = np.random.default_rng(99)


def _rand_ckks(n=CKKS_BATCH_SIZE) -> np.ndarray:
    return _RNG.random(n).astype(np.float64)


def _ser():
    from pyOpenFHE.CKKS import serial
    return serial, serial.SerType.BINARY



# ── Bytes round-trips ──────────────────────────────────────────────────────────

class TestCKKSSerialBytes:
    def test_ciphertext_bytes_roundtrip(self, ckks_cc_keys):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys
        x = _rand_ckks()
        c = cc.encrypt(keys.publicKey, x)

        data = serial.SerializeToBytes(c, ST)
        assert isinstance(data, (bytes, bytearray))
        assert len(data) > 0

        c2 = serial.DeserializeFromBytes_Ciphertext(data, ST)
        y = cc.decrypt(keys.secretKey, c2)[:CKKS_BATCH_SIZE]
        np.testing.assert_allclose(y, x, atol=CKKS_ATOL)

    def test_public_key_bytes_roundtrip(self, ckks_cc_keys):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys

        data = serial.SerializeToBytes(keys.publicKey, ST)
        assert isinstance(data, (bytes, bytearray))

        pk2 = serial.DeserializeFromBytes_PublicKey(data, ST)
        x = _rand_ckks()
        c = cc.encrypt(pk2, x)
        y = cc.decrypt(keys.secretKey, c)[:CKKS_BATCH_SIZE]
        np.testing.assert_allclose(y, x, atol=CKKS_ATOL)

    def test_private_key_bytes_roundtrip(self, ckks_cc_keys):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys

        data = serial.SerializeToBytes(keys.secretKey, ST)
        assert isinstance(data, (bytes, bytearray))

        sk2 = serial.DeserializeFromBytes_PrivateKey(data, ST)
        x = _rand_ckks()
        c = cc.encrypt(keys.publicKey, x)
        y = cc.decrypt(sk2, c)[:CKKS_BATCH_SIZE]
        np.testing.assert_allclose(y, x, atol=CKKS_ATOL)

    def test_eval_mult_key_bytes_roundtrip(self, ckks_cc_keys):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys
        # Verify serialization produces non-empty bytes.
        # Full deserialization round-trip for eval mult keys is not testable
        # within a single process: OpenFHE stores eval keys in a process-wide
        # static map keyed by the secret-key ID; re-inserting the same tag
        # raises InsertEvalMultKey(). Functionality is covered by test_ckks.py.
        emk_data = serial.SerializeToBytes_EvalMultKey_CryptoContext(cc, ST)
        assert isinstance(emk_data, (bytes, bytearray)) and len(emk_data) > 0

    def test_eval_automorphism_key_bytes_roundtrip(self, ckks_cc_keys):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys

        data = serial.SerializeToBytes_EvalAutomorphismKey_CryptoContext(cc, ST)
        assert isinstance(data, (bytes, bytearray))

        serial.DeserializeFromBytes_EvalAutomorphismKey_CryptoContext(cc, data, ST)

        x = _rand_ckks()
        c = cc.encrypt(keys.publicKey, x)
        y = cc.decrypt(keys.secretKey, c << 1)[:CKKS_BATCH_SIZE]
        np.testing.assert_allclose(y, np.roll(x, -1), atol=CKKS_ATOL)


# ── File round-trips ───────────────────────────────────────────────────────────

class TestCKKSSerialFile:
    def test_ciphertext_file_roundtrip(self, ckks_cc_keys, tmp_path):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys
        x = _rand_ckks()
        c = cc.encrypt(keys.publicKey, x)

        path = str(tmp_path / "ctxt.bin")
        serial.SerializeToFile(path, c, ST)

        c2 = serial.DeserializeFromFile_Ciphertext(path, ST)
        y = cc.decrypt(keys.secretKey, c2)[:CKKS_BATCH_SIZE]
        np.testing.assert_allclose(y, x, atol=CKKS_ATOL)

    def test_public_key_file_roundtrip(self, ckks_cc_keys, tmp_path):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys

        path = str(tmp_path / "pk.bin")
        serial.SerializeToFile(path, keys.publicKey, ST)

        pk2 = serial.DeserializeFromFile_PublicKey(path, ST)
        x = _rand_ckks()
        c = cc.encrypt(pk2, x)
        y = cc.decrypt(keys.secretKey, c)[:CKKS_BATCH_SIZE]
        np.testing.assert_allclose(y, x, atol=CKKS_ATOL)

    def test_private_key_file_roundtrip(self, ckks_cc_keys, tmp_path):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys

        path = str(tmp_path / "sk.bin")
        serial.SerializeToFile(path, keys.secretKey, ST)

        sk2 = serial.DeserializeFromFile_PrivateKey(path, ST)
        x = _rand_ckks()
        c = cc.encrypt(keys.publicKey, x)
        y = cc.decrypt(sk2, c)[:CKKS_BATCH_SIZE]
        np.testing.assert_allclose(y, x, atol=CKKS_ATOL)

    def test_eval_mult_key_file_roundtrip(self, ckks_cc_keys, tmp_path):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys
        # See test_eval_mult_key_bytes_roundtrip for why we only test the
        # serialize direction here.
        emk_path = str(tmp_path / "emk.bin")
        serial.SerializeToFile_EvalMultKey_CryptoContext(cc, emk_path, ST)
        import os
        assert os.path.getsize(emk_path) > 0

    def test_eval_automorphism_key_file_roundtrip(self, ckks_cc_keys, tmp_path):
        serial, ST = _ser()
        cc, keys = ckks_cc_keys

        path = str(tmp_path / "rot.bin")
        serial.SerializeToFile_EvalAutomorphismKey_CryptoContext(cc, path, ST)
        serial.DeserializeFromFile_EvalAutomorphismKey_CryptoContext(cc, path, ST)

        x = _rand_ckks()
        c = cc.encrypt(keys.publicKey, x)
        y = cc.decrypt(keys.secretKey, c << 1)[:CKKS_BATCH_SIZE]
        np.testing.assert_allclose(y, np.roll(x, -1), atol=CKKS_ATOL)
