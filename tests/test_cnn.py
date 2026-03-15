"""Smoke tests: verify the CNN module and OMP helpers are importable and callable."""

import pytest

pytest.importorskip("pyOpenFHE")

pytestmark = pytest.mark.slow


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
