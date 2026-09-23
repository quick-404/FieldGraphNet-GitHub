"""Test NEMESYS BCDG analyzer core functions"""
import numpy as np
from nemesys_gnn4id.nemesys.analyzer import _bit_congruence


class TestBitCongruence:
    def test_identical_bytes(self):
        result = _bit_congruence(np.array([0xFF, 0xFF]))
        assert len(result) == 1
        assert result[0] == 1.0

    def test_opposite_bytes(self):
        result = _bit_congruence(np.array([0xFF, 0x00]))
        assert len(result) == 1
        assert result[0] == 0.0

    def test_single_byte_returns_empty(self):
        result = _bit_congruence(np.array([0x42]))
        assert len(result) == 0

    def test_multiple_bytes(self):
        result = _bit_congruence(np.array([0x00, 0xFF, 0x00]))
        assert len(result) == 2
