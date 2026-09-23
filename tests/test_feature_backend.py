# -*- coding: utf-8 -*-
"""Feature-extractor provenance and strict-backend enforcement in extract_features.

The nfstream and scapy extraction paths are NOT equivalent (~49% cell agreement, 12/82
dimensions never agree, and they do not even extract the same number of flows), so a
silent fallback makes any persisted result uninterpretable. These tests pin the three
NEMESYS_FEATURE_BACKEND modes, every record the recorder can write -- ('nfstream', 'ok'),
('scapy', 'forced by ...'), ('scapy', 'nfstream failed: ...'), ('scapy', 'nfstream
unavailable'), ('nfstream-failed', ...) -- and the rule that a failing call overwrites
the record instead of inheriting the previous call's.

They parse no PCAP: the two extractors are replaced by fakes through monkeypatch.

Mode switching: `FEATURE_BACKEND` and `NFSTREAM_AVAILABLE` are module-level constants
computed at import time, but `extract_features` reads them as module GLOBALS at CALL
time, so patching the module attributes (monkeypatch.setattr) is sufficient. That is
what is used here, rather than monkeypatch.setenv + importlib.reload, because a reload
re-imports torch/nfstream (~17 s per test), re-runs the Npcap preload side effects, and
yields a NEW module object whose `_LAST_BACKEND` is a different dict than the one this
test file imported -- i.e. it would verify a different object than the one the rest of
the process uses. Attribute patching exercises the real code path, is automatically
undone by monkeypatch, and keeps the suite fast.
"""
import os
import pathlib

import pytest

from nemesys_gnn4id.gnn4id import analyzer
from nemesys_gnn4id.gnn4id.analyzer import GNN4IDAnalyzer

SENTINEL = object()


@pytest.fixture
def pcap_path(monkeypatch):
    """A PCAP path that 'exists' without touching the filesystem.

    extract_features only calls os.path.exists() on it (the FileNotFoundError guard),
    so that one call is patched instead of creating a file: nothing is parsed here.
    `tmp_path` was deliberately NOT used -- in this environment pytest's tmp root
    (C:\\Users\\<user>\\AppData\\Local\\Temp\\pytest-of-<user>) exists with an ACL that
    denies even os.scandir, so the tmp_path fixture raises PermissionError before the
    test body runs, unrelated to the code under test.
    """
    path = str(pathlib.Path('no-such-dir') / 'empty.pcap')
    real_exists = os.path.exists

    def fake_exists(candidate):
        return True if str(candidate) == path else real_exists(candidate)

    monkeypatch.setattr(os.path, 'exists', fake_exists)
    return path


@pytest.fixture
def calls():
    """Call counters so 'was scapy called?' is an assertion, not an inference."""
    return {'nfstream': 0, 'scapy': 0}


@pytest.fixture
def subject(calls, monkeypatch):
    """A real analyzer whose two extractors are fakes: nfstream raises, scapy succeeds."""

    def nfstream_dies(*args, **kwargs):
        calls['nfstream'] += 1
        raise RuntimeError('nfstream worker died (simulated)')

    def scapy_returns_sentinel(*args, **kwargs):
        calls['scapy'] += 1
        return SENTINEL

    monkeypatch.setattr(GNN4IDAnalyzer, '_extract_with_nfstream', nfstream_dies)
    monkeypatch.setattr(GNN4IDAnalyzer, '_extract_with_scapy', scapy_returns_sentinel)
    # device='cpu' skips the torch.cuda.is_available() probe; nothing else in
    # extract_features touches __init__ state
    return GNN4IDAnalyzer(device='cpu')


@pytest.fixture
def subject_nfstream_ok(calls, monkeypatch):
    """A real analyzer whose NFSTREAM extractor succeeds (scapy must never be reached)."""

    def nfstream_returns_sentinel(*args, **kwargs):
        calls['nfstream'] += 1
        return SENTINEL

    def scapy_forbidden(*args, **kwargs):
        calls['scapy'] += 1
        raise AssertionError('scapy must not be called when nfstream succeeds')

    monkeypatch.setattr(GNN4IDAnalyzer, '_extract_with_nfstream',
                        nfstream_returns_sentinel)
    monkeypatch.setattr(GNN4IDAnalyzer, '_extract_with_scapy', scapy_forbidden)
    return GNN4IDAnalyzer(device='cpu')


def test_forced_nfstream_raises_and_never_calls_scapy(pcap_path, calls, subject,
                                                      monkeypatch):
    """NEMESYS_FEATURE_BACKEND=nfstream: fail loudly, do NOT degrade to scapy."""
    monkeypatch.setattr(analyzer, 'FEATURE_BACKEND', 'nfstream')
    monkeypatch.setattr(analyzer, 'NFSTREAM_AVAILABLE', True)

    with pytest.raises(RuntimeError) as excinfo:
        subject.extract_features(pcap_path)

    assert calls == {'nfstream': 1, 'scapy': 0}, calls
    message = str(excinfo.value)
    assert 'forbids the scapy fallback' in message
    assert 'NEMESYS_FEATURE_BACKEND=nfstream' in message
    assert 'simulated' in message
    # the original failure must survive as the cause, not be swallowed
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert 'nfstream worker died' in str(excinfo.value.__cause__)

    backend, reason = analyzer.last_feature_backend()
    assert backend == 'nfstream-failed'
    assert 'simulated' in reason


def test_auto_falls_back_to_scapy_and_records_why(pcap_path, calls, subject,
                                                 monkeypatch):
    """NEMESYS_FEATURE_BACKEND=auto: degrade, but record the backend and the reason."""
    monkeypatch.setattr(analyzer, 'FEATURE_BACKEND', 'auto')
    monkeypatch.setattr(analyzer, 'NFSTREAM_AVAILABLE', True)

    assert subject.extract_features(pcap_path) is SENTINEL
    assert calls == {'nfstream': 1, 'scapy': 1}, calls

    backend, reason = analyzer.last_feature_backend()
    assert backend == 'scapy'
    assert 'nfstream failed' in reason
    assert 'simulated' in reason


def test_forced_scapy_short_circuits_before_nfstream(pcap_path, calls, subject,
                                                    monkeypatch):
    """NEMESYS_FEATURE_BACKEND=scapy: go straight to scapy, never attempt nfstream.

    NFSTREAM_AVAILABLE is forced True here (the real module sets it False whenever
    scapy is forced, see analyzer.py line 79) precisely so that this test fails if the
    'scapy forced' short-circuit is ever moved below the nfstream branch: with
    NFSTREAM_AVAILABLE False the two orderings would be indistinguishable.
    """
    monkeypatch.setattr(analyzer, 'FEATURE_BACKEND', 'scapy')
    monkeypatch.setattr(analyzer, 'NFSTREAM_AVAILABLE', True)

    assert subject.extract_features(pcap_path) is SENTINEL
    assert calls == {'nfstream': 0, 'scapy': 1}, calls

    backend, reason = analyzer.last_feature_backend()
    assert backend == 'scapy'
    assert 'forced by NEMESYS_FEATURE_BACKEND=scapy' in reason


def test_successful_nfstream_records_ok(pcap_path, calls, subject_nfstream_ok,
                                        monkeypatch):
    """The success record: ('nfstream', 'ok'), and scapy is never touched.

    Pinned because the success record is the ONLY branch where the backend means 'this
    extractor produced the returned features' (the scapy branches record before the
    call, i.e. 'this extractor was invoked' -- see last_feature_backend()). If this
    record were ever lost, a run's JSON would carry feature_backend_used=None and
    nothing downstream could tell which 82-dim feature set it describes.
    """
    monkeypatch.setattr(analyzer, 'FEATURE_BACKEND', 'auto')
    monkeypatch.setattr(analyzer, 'NFSTREAM_AVAILABLE', True)

    assert subject_nfstream_ok.extract_features(pcap_path) is SENTINEL
    assert calls == {'nfstream': 1, 'scapy': 0}, calls
    assert analyzer.last_feature_backend() == ('nfstream', 'ok')


def test_auto_without_nfstream_records_unavailable_and_uses_scapy(
        pcap_path, calls, subject, monkeypatch):
    """nfstream not installed/usable + auto: degrade to scapy, but say why.

    This is the branch behind the historical '3233 vs 3500 flows' confusion: the
    fallback is legitimate in auto mode and MUST be visible in the record, otherwise a
    result extracted with scapy is indistinguishable from an nfstream one.
    """
    monkeypatch.setattr(analyzer, 'FEATURE_BACKEND', 'auto')
    monkeypatch.setattr(analyzer, 'NFSTREAM_AVAILABLE', False)

    assert subject.extract_features(pcap_path) is SENTINEL
    # nfstream must not even be attempted when it is unavailable
    assert calls == {'nfstream': 0, 'scapy': 1}, calls
    assert analyzer.last_feature_backend() == ('scapy', 'nfstream unavailable')


def test_forced_nfstream_unavailable_raises_instead_of_falling_back(
        pcap_path, calls, subject, monkeypatch):
    """NEMESYS_FEATURE_BACKEND=nfstream + nfstream unusable: raise, do NOT fall back.

    Before this was pinned, extract_features recorded ('scapy', 'nfstream unavailable')
    and returned scapy features -- contradicting the error message of the sibling
    branch ('forbids the scapy fallback'). A forced-nfstream job that silently ran on
    scapy is exactly the mixed-backend hazard the provenance record exists to prevent.
    """
    monkeypatch.setattr(analyzer, 'FEATURE_BACKEND', 'nfstream')
    monkeypatch.setattr(analyzer, 'NFSTREAM_AVAILABLE', False)

    with pytest.raises(RuntimeError) as excinfo:
        subject.extract_features(pcap_path)

    assert calls == {'nfstream': 0, 'scapy': 0}, calls
    message = str(excinfo.value)
    assert 'nfstream is unavailable' in message
    assert 'NEMESYS_FEATURE_BACKEND=nfstream' in message
    assert 'forbids the scapy fallback' in message

    backend, reason = analyzer.last_feature_backend()
    assert backend == 'nfstream-failed'
    assert 'unavailable' in reason


def test_failed_call_does_not_inherit_the_previous_record(pcap_path, calls,
                                                          subject_nfstream_ok,
                                                          monkeypatch):
    """A failing call overwrites the record; it never leaves the last call's behind.

    The FIRST call here succeeds on nfstream ('nfstream', 'ok'). The second fails on
    nfstream with the backend forced, so a caller that catches the RuntimeError and then
    reads last_feature_backend() must not see the stale success -- that stale read would
    be written into a result JSON as feature_backend_used='nfstream'.
    """
    monkeypatch.setattr(analyzer, 'FEATURE_BACKEND', 'nfstream')
    monkeypatch.setattr(analyzer, 'NFSTREAM_AVAILABLE', True)

    assert subject_nfstream_ok.extract_features(pcap_path) is SENTINEL
    assert analyzer.last_feature_backend() == ('nfstream', 'ok')

    def nfstream_dies_now(*args, **kwargs):
        calls['nfstream'] += 1
        raise RuntimeError('second call died (simulated)')

    monkeypatch.setattr(GNN4IDAnalyzer, '_extract_with_nfstream', nfstream_dies_now)

    with pytest.raises(RuntimeError):
        subject_nfstream_ok.extract_features(pcap_path)

    backend, reason = analyzer.last_feature_backend()
    assert (backend, reason) != ('nfstream', 'ok')
    assert backend == 'nfstream-failed'
    assert 'second call died' in reason
    assert calls == {'nfstream': 2, 'scapy': 0}, calls
