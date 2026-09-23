"""Tests for the CICIDS2017 schedule module (time-window labelling)."""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))

import cicids2017_schedule as S


def test_chunk_name_is_utc_plus_8():
    # filename 19:59:39 local(UTC+8) -> 11:59:39 UTC
    assert S.utc_from_chunk_name('Fri_chunk_00000_20170707195939.pcap') == \
        dt.datetime(2017, 7, 7, 11, 59, 39)
    assert S.utc_from_chunk_name('Wed_chunk_00000_20170705194242.pcap') == \
        dt.datetime(2017, 7, 5, 11, 42, 42)
    assert S.utc_from_chunk_name('no_timestamp_here.pcap') is None


def test_local_offset_is_adt():
    # UNB documentation: capture ran 09:00-17:00 local; ADT = UTC-3.
    assert S.LOCAL_TO_UTC_OFFSET_HOURS == -3


def test_window_for_inside_and_outside():
    # Friday Botnet ARES 10:02-11:02 ADT == 13:02-14:02 UTC
    assert S.window_for(dt.datetime(2017, 7, 7, 13, 30)) == 'Botnet_ARES'
    # 09:30 ADT is before any Friday attack
    assert S.window_for(dt.datetime(2017, 7, 7, 12, 30)) == 'Benign'
    # 16:00 ADT falls in DDoS LOIT (15:56-16:16)
    assert S.window_for(dt.datetime(2017, 7, 7, 19, 0)) == 'DDoS_LOIT'


def test_window_boundaries_are_inclusive():
    # DoS_Hulk is 10:43-11:00 ADT == 13:43-14:00 UTC
    assert S.window_for(dt.datetime(2017, 7, 5, 13, 43)) == 'DoS_Hulk'
    assert S.window_for(dt.datetime(2017, 7, 5, 14, 0)) == 'DoS_Hulk'
    assert S.window_for(dt.datetime(2017, 7, 5, 13, 42)) == 'Benign'
    assert S.window_for(dt.datetime(2017, 7, 5, 14, 1)) == 'Benign'


def test_margin_extends_windows():
    just_before = dt.datetime(2017, 7, 5, 13, 42)   # 1 min before DoS_Hulk
    assert S.window_for(just_before) == 'Benign'
    assert S.window_for(just_before, margin_minutes=1) == 'DoS_Hulk'


def test_windows_overlapping_detects_multi_window_chunk():
    # a chunk spanning 09:40-10:20 ADT touches slowloris (9:47-10:10) and
    # slowhttptest (10:14-10:35)
    lo = dt.datetime(2017, 7, 5, 12, 40)   # 09:40 ADT
    hi = dt.datetime(2017, 7, 5, 13, 20)   # 10:20 ADT
    got = S.windows_overlapping(lo, hi)
    assert 'DoS_Slowloris' in got
    assert 'DoS_Slowhttptest' in got
    assert 'DoS_Hulk' not in got


def test_every_window_falls_inside_a_measured_capture_span():
    # Packet-timestamp spans measured with scapy (naive UTC).
    spans = {
        5: (dt.datetime(2017, 7, 5, 11, 42, 42), dt.datetime(2017, 7, 5, 20, 6, 31)),
        7: (dt.datetime(2017, 7, 7, 11, 59, 50), dt.datetime(2017, 7, 7, 20, 1, 22)),
    }
    off = dt.timedelta(hours=S.LOCAL_TO_UTC_OFFSET_HOURS)
    for date_s, label, a, b, _n in S.WINDOWS:
        d = dt.date.fromisoformat(date_s)
        lo = dt.datetime.combine(d, dt.time(*map(int, a.split(':')))) - off
        hi = dt.datetime.combine(d, dt.time(*map(int, b.split(':')))) - off
        s, e = spans[d.day]
        assert s <= lo and hi <= e, f'{label} falls outside the measured capture'


def test_schedule_records_its_source():
    # the labelling must be auditable: source URL, retrieval date and citation
    assert S.SOURCE_URL.startswith('https://www.unb.ca/')
    assert S.RETRIEVED
    assert 'Sharafaldin' in S.CITATION
