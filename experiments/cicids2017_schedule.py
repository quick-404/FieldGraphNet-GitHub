# -*- coding: utf-8 -*-
"""Authoritative CICIDS2017 attack schedule, for time-window labelling of the pcaps.

WHY THIS FILE EXISTS
--------------------
The CICIDS2017 pcaps (``data/data/17pcap``) contain a mix of benign traffic and
several attacks inside a single file, and the accompanying MachineLearningCVE CSVs
in this repository have had their **identity columns stripped** (they start at
"Destination Port"; there is no Flow ID / Source IP / Source Port / Destination IP /
Protocol / Timestamp). Row-order alignment between CSV and pcap is therefore
impossible, and a previous task correctly recorded that as INFEASIBLE.

It is, however, NOT necessary: each attack was executed in a documented time window,
the pcap chunks carry their capture time in the filename, and the schedule is
published by the dataset authors. Labelling by time window is thus possible without
any CSV alignment. This module records that schedule so the labelling is auditable
rather than reconstructed from memory.

SOURCE (retrieved 2026-09-22)
-----------------------------
    https://www.unb.ca/cic/datasets/ids-2017.html   ("Intrusion Detection Evaluation Dataset (CIC-IDS2017)")
    Original paper: Sharafaldin I., Habibi Lashkari A., Ghorbani A. A.,
    "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
    Characterization", ICISSP 2018, Portugal.  <-- cite this when using the dataset.

TIMEBASE (verified, not assumed)
--------------------------------
* The **packet timestamps are UTC**.  ``scapy`` reports them as epoch seconds.
* The **chunk filenames** encode this machine's local time at chunking time,
  i.e. **UTC+8**: e.g. ``Fri_chunk_00000_20170707195939.pcap`` holds a first packet
  at ``2017-07-07 11:59:50 UTC`` -- an 8-hour offset.  Verified on three chunks
  (Fri first/last and Wed first).
* The **schedule times below are in the capture site's local time**, which for UNB
  in July 2017 is **ADT = UTC-3**.  This is not guessed: the dataset documentation
  states the capture ran 09:00 Monday to 17:00 Friday, and converting our measured
  spans to ADT reproduces exactly that:
      Friday    : 11:59:50 - 20:01:22 UTC  ->  08:59:50 - 17:01:22 ADT
      Wednesday : 11:42:42 - 20:06:31 UTC  ->  08:42:42 - 17:06:31 ADT

CONSEQUENCE: to label a chunk, take its packet timestamp, subtract 3 hours to get
ADT, and compare with the windows below.  Use :func:`window_for`.
"""
import datetime as dt

SOURCE_URL = 'https://www.unb.ca/cic/datasets/ids-2017.html'
RETRIEVED = '2026-09-22'
CITATION = ('Sharafaldin I., Habibi Lashkari A., Ghorbani A. A. "Toward Generating '
            'a New Intrusion Detection Dataset and Intrusion Traffic '
            'Characterization", ICISSP 2018, Portugal.')

# Local time is ADT (UTC-3) for the July 2017 capture.
LOCAL_TO_UTC_OFFSET_HOURS = -3

# Attack windows in LOCAL time (ADT). Each entry:
#   (date, label, start "HH:MM", end "HH:MM", note)
# Times are exactly as published by UNB; do not round or "tidy" them.
WINDOWS = [
    # ---- Wednesday, 5 July 2017 -- DoS / Heartbleed ----
    ('2017-07-05', 'DoS_Slowloris',     '09:47', '10:10', ''),
    ('2017-07-05', 'DoS_Slowhttptest',  '10:14', '10:35', ''),
    ('2017-07-05', 'DoS_Hulk',          '10:43', '11:00', 'published as "11 a.m."'),
    ('2017-07-05', 'DoS_GoldenEye',     '11:10', '11:23', ''),
    ('2017-07-05', 'Heartbleed',        '15:12', '15:32', 'port 444'),
    # ---- Friday, 7 July 2017 -- Botnet / PortScan / DDoS ----
    ('2017-07-07', 'Botnet_ARES',       '10:02', '11:02', ''),
    # PortScan was run in many short bursts, first behind firewall rules and then
    # with the rules off. Rather than 27 tiny windows we record the two phases;
    # the gaps between bursts are short (<= 8 min) relative to a ~14 s chunk.
    ('2017-07-07', 'PortScan_rules_on', '13:55', '14:35', '12 bursts, see UNB page'),
    ('2017-07-07', 'PortScan_rules_off', '14:51', '15:29', '15 bursts, see UNB page'),
    ('2017-07-07', 'DDoS_LOIT',         '15:56', '16:16', ''),
]

# Days present in data/data/17pcap and fully covered by the windows above.
CAPTURES = {
    'Wed': 'Wednesday-workingHours.pcap',
    'Fri': 'Friday-WorkingHours.pcap',
}


def _to_local(moment):
    """UTC datetime -> capture-site local (ADT)."""
    return moment + dt.timedelta(hours=LOCAL_TO_UTC_OFFSET_HOURS)


def window_for(utc_moment, margin_minutes=0):
    """Return the attack label whose window contains ``utc_moment``, else 'Benign'.

    ``utc_moment`` is a naive UTC datetime (what ``scapy`` gives you via
    ``datetime.utcfromtimestamp(pkt.time)``).

    ``margin_minutes`` extends each window symmetrically; use it for the boundary
    sensitivity check (the paper reports the conclusion must not depend on it).
    A chunk overlapping several windows is assigned the FIRST match in WINDOWS
    order; callers that need to know about overlap should use
    :func:`windows_overlapping` instead.
    """
    local = _to_local(utc_moment)
    for date_s, label, start_s, end_s, _note in WINDOWS:
        d = dt.date.fromisoformat(date_s)
        if local.date() != d:
            continue
        sh, sm = (int(x) for x in start_s.split(':'))
        eh, em = (int(x) for x in end_s.split(':'))
        lo = local.replace(hour=sh, minute=sm, second=0, microsecond=0) \
            - dt.timedelta(minutes=margin_minutes)
        hi = local.replace(hour=eh, minute=em, second=0, microsecond=0) \
            + dt.timedelta(minutes=margin_minutes)
        if lo <= local <= hi:
            return label
    return 'Benign'


def windows_overlapping(start_utc, end_utc, margin_minutes=0):
    """Every attack label whose window overlaps [start_utc, end_utc] (a chunk span)."""
    hits = []
    ls, le = _to_local(start_utc), _to_local(end_utc)
    for date_s, label, start_s, end_s, _note in WINDOWS:
        d = dt.date.fromisoformat(date_s)
        sh, sm = (int(x) for x in start_s.split(':'))
        eh, em = (int(x) for x in end_s.split(':'))
        lo = dt.datetime.combine(d, dt.time(sh, sm)) - dt.timedelta(minutes=margin_minutes)
        hi = dt.datetime.combine(d, dt.time(eh, em)) + dt.timedelta(minutes=margin_minutes)
        if ls <= hi and le >= lo:
            hits.append(label)
    return hits


def utc_from_chunk_name(chunk_name):
    """UTC datetime of the capture time encoded in a 17pcap chunk filename.

    ``Fri_chunk_00000_20170707195939.pcap`` -> ``2017-07-07 11:59:39 UTC``
    (the filename is UTC+8, see the module docstring).  Returns None if the name
    does not carry a timestamp.
    """
    import re
    m = re.search(r'_(\d{14})\.pcap$', chunk_name)
    if not m:
        return None
    local = dt.datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
    return local - dt.timedelta(hours=8)


if __name__ == '__main__':
    print(f'schedule source : {SOURCE_URL} (retrieved {RETRIEVED})')
    print(f'local timebase  : ADT = UTC{LOCAL_TO_UTC_OFFSET_HOURS:+d}')
    print(f'{len(WINDOWS)} attack windows across {len(CAPTURES)} captures\n')
    for date_s, label, start_s, end_s, note in WINDOWS:
        d = dt.date.fromisoformat(date_s)
        lo = dt.datetime.combine(d, dt.time(*map(int, start_s.split(':'))))
        hi = dt.datetime.combine(d, dt.time(*map(int, end_s.split(':'))))
        print(f'  {date_s}  {label:<20} {start_s}-{end_s} local'
              f'  |  UTC {lo - dt.timedelta(hours=LOCAL_TO_UTC_OFFSET_HOURS)}'
              f' - {hi - dt.timedelta(hours=LOCAL_TO_UTC_OFFSET_HOURS)}'
              + (f'   ({note})' if note else ''))
