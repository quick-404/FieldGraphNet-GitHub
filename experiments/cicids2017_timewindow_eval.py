# -*- coding: utf-8 -*-
"""CICIDS2017 time-window feasibility pilot: chunk-level vs attacker-IP flow-level labels.

WHY
---
The MachineLearningCVE CSVs in this repo have had their identity columns stripped, so
CSV<->pcap alignment is INFEASIBLE (experiments/eval_cicids2017_labels.py).  But each
attack was executed in a *documented time window* and the chunk filenames carry the
capture time, so labels can be derived without any CSV (experiments/cicids2017_schedule.py
records and verifies that schedule; this file does NOT re-derive or modify it).

This pilot asks whether that route is defensible enough to carry a second public dataset
in the paper, and at which granularity:

  Scheme A (chunk level)   a chunk is labelled by window_for(chunk_start_utc); a chunk
                           "detects" the attack if >=1 of its flows alerts.
  Scheme B (flow level)    inside a window, a flow is an ATTACK flow iff it involves the
                           dataset authors' published Kali attacker 205.174.165.73, else
                           benign; outside every window every flow is benign.  This gives
                           genuine per-flow labels, so it is far more valuable -- but only
                           if that IP really appears in the window's flows and does NOT
                           also carry ordinary traffic.  BOTH are measured here.

SCOPE (Wednesday 2017-07-05 only: 5 DoS/Heartbleed windows, all in the morning UTC)
  20 chunks per window x 5 windows = 100 attack chunks
  100 benign chunks drawn from outside every window and >= BENIGN_MARGIN_MINUTES away
  from any window boundary, so window spillover cannot contaminate them.
  Total 200 chunks, one pipeline run each, sampled deterministically and recorded.

WHAT IS REUSED, NOT REINVENTED
  * the schedule and its timebase : experiments/cicids2017_schedule.py (untouched)
  * the pipeline driver           : experiments/run_ablation_fusion.py (max_flows=50,
                                    same model/proto-model defaults)
  * metric arithmetic             : run_ablation_fusion.binary_metrics
  * controlled-FPR convention     : the tie-aware benign-only threshold rule of
                                    experiments/eval_ood_method_comparison.py
                                    :threshold_at_target_fpr, and the rule-based variant
                                    of it that experiments/ablation_controlled_fpr.py
                                    uses for the A4 (whole fusion rule) row -- candidates
                                    = distinct benign scores, operating point = closest
                                    realised FPR, detection strict ">".  Copied verbatim
                                    in semantics; nothing new is invented.
  * per-flow scores               : the pipeline's ALREADY-EXISTING gated diagnostic
                                    (NEMESYS_ABLATION_DUMP=1 -> r['_diag']).  No pipeline
                                    code is added or changed by this file: flow metadata
                                    (src/dst IP+port, protocol) is already returned in
                                    result['flow_metadata'] and each result carries
                                    flow_index.  --verify-inert proves the flag is inert.

HONEST REPORTING
  Negative results are first-class here: if 205.174.165.73 never shows up in the sampled
  flows, Scheme B is not viable and the JSON says so instead of reporting fabricated
  flow-level numbers.  Nothing is tuned to improve any number.

USAGE
  $env:NEMESYS_FEATURE_BACKEND='scapy'      # nfstream cannot open this repo's non-ASCII
                                            # path (its C layer mangles it) nor IPv6 flows
  .venv\\Scripts\\python.exe experiments/cicids2017_timewindow_eval.py [--dry-run]
      [--limit-per-group N] [--dump|--no-dump] [--max-flows 50] [--out PATH]
      [--per-flow-out PATH] [--seed 42]
  .venv\\Scripts\\python.exe experiments/cicids2017_timewindow_eval.py \\
      --verify-inert A_per_flow.json B_per_flow.json      # flag on vs off comparison
"""
import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, HERE)

from cicids2017_schedule import (CITATION, LOCAL_TO_UTC_OFFSET_HOURS, SOURCE_URL,
                                 WINDOWS, utc_from_chunk_name, window_for,
                                 windows_overlapping)

CHUNK_DIR = os.path.join(REPO, 'data', 'data', '17pcap_chunks')
EVAL_DIR = os.path.join(REPO, 'eval_results', 'fusion_comparison')

DAY_PREFIX = 'Wed'
DAY_DATE = '2017-07-05'

# The two days present in data/data/17pcap_chunks. Wednesday carries the DoS family
# (Slowloris/Slowhttptest/Hulk/GoldenEye) plus Heartbleed; Friday carries the Botnet,
# PortScan and DDoS families that Wednesday does not have at all. Both are selected by
# --day, which rebinds DAY_PREFIX/DAY_DATE before any chunk enumeration happens.
DAY_CHOICES = {
    'Wed': ('Wed', '2017-07-05'),
    'Fri': ('Fri', '2017-07-07'),
}
N_PER_WINDOW = 20
N_BENIGN = 100
BENIGN_MARGIN_MINUTES = 10
SAMPLE_SEED = 42

# The dataset authors' published attacker host for CICIDS2017 (Kali).
ATTACKER_IP = '205.174.165.73'

# Addresses the dataset documentation/literature associates with the CICIDS2017 testbed
# (attackers, the public victim web server, the internal victims/server, the gateway).
# Scanned by --byte-scan purely to CHARACTERISE what the local capture actually contains;
# none of them is ever used as a label, so this is diagnosis, not tuning.
CANDIDATE_IPS = [
    '205.174.165.73',   # published Kali attacker (the Scheme B marker)
    '205.174.165.68',   # published public address of the victim web server
    '205.174.165.80', '205.174.165.81', '205.174.165.82',
    '172.16.0.1',       # gateway seen in the local capture
    '192.168.10.50',    # internal web server / NAT
]

# Copied from nemesys_gnn4id.pipeline / experiments/run_ablation_fusion.py.
FUSION_ALERT_STATUSES = {
    'confirmed_gnn_attack',
    'suspicious_unknown_protocol_attack',
    'possible_missed_unknown_protocol_attack',
}

MODEL_PATH = 'models/model.pth'
PROTO_MODEL_DEFAULT = 'models/field_proto_model_v2.pth'
DEFAULT_MAX_FLOWS = 50

# The project's controlled-FPR target (eval_ood_method_comparison.TARGET_FPR).
TARGET_FPR = 0.20
MARGINS = (-1, 0, 1)

DEFAULT_OUT = os.path.join(EVAL_DIR, 'cicids2017_timewindow_pilot.json')
DEFAULT_PER_FLOW_OUT = os.path.join(EVAL_DIR,
                                    'cicids2017_timewindow_pilot_per_flow.json')


# ---------------------------------------------------------------------------
# metric arithmetic -- run_ablation_fusion.binary_metrics is the repo convention
# ---------------------------------------------------------------------------
def _raf():
    import run_ablation_fusion
    return run_ablation_fusion


def threshold_at_target_fpr(benign_scores, target_fpr=TARGET_FPR):
    """Benign-only threshold whose REALISED FPR is closest to target_fpr.

    Copied verbatim in semantics from
    experiments/eval_ood_method_comparison.py:threshold_at_target_fpr (and re-copied in
    experiments/ablation_controlled_fpr.py): candidates are each distinct benign score
    plus one endpoint just above the largest; ties are broken towards the closest
    realised FPR (first minimum wins while iterating ascending thresholds, i.e. the
    smallest threshold among ties); detection is a strict ">".
    """
    import numpy as np
    b = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(b)
    if n == 0:
        return float('inf')
    best_thr, best_gap = float('inf'), float('inf')
    for t in np.append(np.unique(b), b[-1] + 1e-12):
        gap = abs(float((b > t).sum()) / n - target_fpr)
        if gap < best_gap - 1e-12:
            best_thr, best_gap = float(t), gap
    return best_thr


def select_operating_point(fprs, target_fpr=TARGET_FPR):
    """Index of the candidate whose realised FPR is closest to target_fpr.

    Same strict-improvement tie resolution as threshold_at_target_fpr (ascending
    candidates -> smallest threshold among ties).  Copied semantics of
    experiments/ablation_controlled_fpr.py:select_operating_point.
    """
    best_i, best_gap = 0, float('inf')
    for i, f in enumerate(fprs):
        gap = abs(float(f) - target_fpr)
        if gap < best_gap - 1e-12:
            best_i, best_gap = i, gap
    return best_i


def replay_a4_alert(is_attack, in_domain, known_proto, cluster_high):
    """`FUSION_ALERT_STATUSES` membership of _build_fusion_decision's branch table.

    Literal transcription of the branch conditions in
    src/nemesys_gnn4id/pipeline.py:_build_fusion_decision, reduced to alert-vs-not
    (identical to experiments/ablation_controlled_fpr.py:replay_a4_alerts).  Used only to
    move the fusion RULE along the cluster_score axis, because the rule itself has no
    scalar score of its own.
    """
    if is_attack and in_domain:
        return 'confirmed_gnn_attack' in FUSION_ALERT_STATUSES
    if is_attack and not in_domain and (not known_proto) and cluster_high:
        return 'suspicious_unknown_protocol_attack' in FUSION_ALERT_STATUSES
    if is_attack and not in_domain:
        return 'ood_gnn_possible_false_positive' in FUSION_ALERT_STATUSES
    if (not is_attack) and (not in_domain) and (not known_proto) and cluster_high:
        return 'possible_missed_unknown_protocol_attack' in FUSION_ALERT_STATUSES
    if (not known_proto) and (not in_domain):
        return 'unknown_protocol_monitor' in FUSION_ALERT_STATUSES
    return 'low_risk_known_protocol' in FUSION_ALERT_STATUSES


def _counts(y, pred):
    import numpy as np
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=bool)
    return (int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum()),
            int(((y == 1) & ~pred).sum()), int(((y == 0) & ~pred).sum()))


def _rates(tp, fp, fn, tn):
    return {'recall': tp / max(tp + fn, 1), 'precision': tp / max(tp + fp, 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'fpr': fp / max(fp + tn, 1)}


def _round_rates(rt):
    return {k: round(v, 6) for k, v in rt.items()}


# ---------------------------------------------------------------------------
# chunk enumeration / sampling
# ---------------------------------------------------------------------------
def chunk_index(name):
    tail = name.rsplit('chunk', 1)[-1]
    return int(tail.split('_')[1])


def enumerate_chunks():
    """Wednesday chunks in capture order, with their filename timebase and label.

    Each entry: {name, path, utc (from filename), local (ADT), label_at_0, duration_s}.
    ``duration_s`` is estimated as the gap to the NEXT chunk's start (the chunks were
    split by packet count, so the gap varies); the last chunk reuses the median gap.
    It is needed for the boundary argument: a benign chunk whose start sits >= 10 min
    from every boundary cannot reach a window only if its own span is < 10 min."""
    names = [n for n in os.listdir(CHUNK_DIR)
             if n.startswith(f'{DAY_PREFIX}_chunk_') and n.endswith('.pcap')]
    names.sort(key=chunk_index)
    out = []
    unparsed = []
    for n in names:
        utc = utc_from_chunk_name(n)
        if utc is None:
            unparsed.append(n)
            continue
        out.append({
            'name': n,
            'path': os.path.join(CHUNK_DIR, n),
            'index': chunk_index(n),
            'utc': utc,
            'local': utc + dt.timedelta(hours=LOCAL_TO_UTC_OFFSET_HOURS),
            'label_at_0': window_for(utc),
        })
    gaps = [(out[i + 1]['utc'] - out[i]['utc']).total_seconds() for i in range(len(out) - 1)]
    gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    for i, c in enumerate(out):
        c['duration_s'] = gaps[i] if i < len(gaps) else gap
    return out, unparsed, {'n_chunks': len(out), 'median_gap_s': gap,
                           'max_gap_s': max(gaps) if gaps else 0.0,
                           'min_gap_s': min(gaps) if gaps else 0.0,
                           'n_zero_gap': sum(1 for g in gaps if g == 0.0)}


def day_windows():
    """The schedule's windows for the pilot day, in WINDOWS order (never reordered)."""
    return [(d, label, s, e, note) for (d, label, s, e, note) in WINDOWS if d == DAY_DATE]


def select_sample(chunks, seed=SAMPLE_SEED):
    """Deterministic sample: N_PER_WINDOW per window + N_BENIGN benign chunks."""
    rng = random.Random(seed)
    per_window, attack_pools = {}, {}
    for _d, label, _s, _e, _n in day_windows():
        pool = [c for c in chunks if c['label_at_0'] == label]
        attack_pools[label] = [c['name'] for c in pool]
        if len(pool) < N_PER_WINDOW:
            raise RuntimeError(f'window {label} has only {len(pool)} chunks, '
                               f'need {N_PER_WINDOW}')
        chosen = sorted((c['name'] for c in rng.sample(pool, N_PER_WINDOW)),
                        key=chunk_index)
        per_window[label] = chosen

    benign_pool = [c for c in chunks
                   if window_for(c['utc'], BENIGN_MARGIN_MINUTES) == 'Benign']
    if len(benign_pool) < N_BENIGN:
        raise RuntimeError(f'benign pool has only {len(benign_pool)} chunks, '
                           f'need {N_BENIGN}')
    benign = sorted((c['name'] for c in rng.sample(benign_pool, N_BENIGN)),
                    key=chunk_index)

    attack = [n for _d, label, _s, _e, _n in day_windows() for n in per_window[label]]
    assert not (set(attack) & set(benign)), 'sample groups overlap'
    return {'per_window': per_window, 'attack': attack, 'benign': benign,
            'attack_pool_sizes': {k: len(v) for k, v in attack_pools.items()},
            'benign_pool_size': len(benign_pool),
            'benign_pool_margin_minutes': BENIGN_MARGIN_MINUTES}


def span_overlap_audit(chunks, sample, span_info):
    """Does any SAMPLED chunk straddle a window boundary despite the margin?

    Uses windows_overlapping(start, start + duration_s) per sampled chunk.  For the
    benign group this must be empty on every chunk: that is what ``margin_minutes=10``
    buys, and it is only sound because the longest chunk gap (``max_gap_s``) is far below
    10 minutes.  For attack chunks, >1 overlapping label would mean the chunk sits on a
    boundary between two windows.
    """
    lookup = {c['name']: c for c in chunks}

    def audit(names):
        multi, none, unclear = [], [], 0
        for n in names:
            c = lookup[n]
            end = c['utc'] + dt.timedelta(seconds=c['duration_s'])
            hits = windows_overlapping(c['utc'], end)
            if len(hits) > 1:
                multi.append({'chunk': n, 'labels': hits})
            if not hits:
                none.append(n)
            if c['duration_s'] <= 0:
                unclear += 1
        return multi, none, unclear

    b_multi, b_none, b_unclear = audit(sample['benign'])
    a_multi, a_none, a_unclear = audit(sample['attack'])
    return {
        'rule': 'windows_overlapping(chunk_start_utc, chunk_start_utc + duration_s)',
        'chunk_gaps': span_info,
        'benign_margin_minutes': BENIGN_MARGIN_MINUTES,
        'margin_exceeds_longest_chunk': span_info['max_gap_s'] < BENIGN_MARGIN_MINUTES * 60,
        'benign_chunks_sampled': len(sample['benign']),
        'benign_chunks_overlapping_any_window': len(sample['benign']) - len(b_none),
        'benign_chunks_without_any_window_overlap': len(b_none),
        'benign_chunks_overlapping_two_windows': len(b_multi),
        'attack_chunks_sampled': len(sample['attack']),
        'attack_chunks_overlapping_two_windows': len(a_multi),
        'chunks_with_zero_estimated_duration': b_unclear + a_unclear,
        'attack_chunk_overlaps': a_multi[:10],
        'note': ('benign_chunks_overlapping_any_window must be 0 for the benign sample to be '
                 'spillover-free; chunks_with_zero_estimated_duration are chunks whose '
                 'consecutive filenames share a second, so their span is not resolvable from '
                 'the names alone'),
    }


def raw_ip_scan(path):
    """Stream the RAW packets of one chunk: how much of it involves ATTACKER_IP.

    Independent of the pipeline and of max_flows truncation, so it separates "the marker
    is not in this capture" from "the marker was cut off by df.head(max_flows)".  Flows
    are the same bidirectional 5-tuple grouping the scapy extractor uses, in first-packet
    order, so ``attacker_first_flow_rank`` is directly comparable with max_flows: a rank
    greater than max_flows means the pipeline never saw that flow.

    It also records the top talkers by packet count.  That is diagnosis, not tuning: if
    the published attacker address is absent, the question "which endpoints DO carry this
    window's traffic" has to be answerable from the data rather than guessed.
    """
    from collections import Counter
    from scapy.utils import PcapReader
    from scapy.layers.inet import IP, TCP, UDP
    n_pkts = n_ip = n_att_pkts = 0
    flow_pkts = {}                      # ordered by first packet (dict keeps order)
    flow_attacker = set()
    ip_counter = Counter()
    prefix_counter = Counter()
    r = PcapReader(path)
    for pkt in r:
        n_pkts += 1
        if IP not in pkt:
            continue
        n_ip += 1
        proto = pkt[IP].proto
        sport = dport = 0
        if TCP in pkt:
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
        src, dst = pkt[IP].src, pkt[IP].dst
        ip_counter[src] += 1
        for ip in (src, dst):
            prefix_counter['.'.join(ip.split('.')[:3]) + '.0/24'] += 1
        left, right = (src, sport), (dst, dport)
        key = ((left + right) if left <= right else (right + left)) + (proto,)
        flow_pkts[key] = flow_pkts.get(key, 0) + 1
        if ATTACKER_IP in (src, dst):
            n_att_pkts += 1
            flow_attacker.add(key)
    r.close()
    rank = None
    for i, key in enumerate(flow_pkts, 1):
        if key in flow_attacker:
            rank = i
            break
    top_flows = sorted(flow_pkts.items(), key=lambda kv: -kv[1])[:3]
    return {
        'n_packets': n_pkts,
        'n_ip_packets': n_ip,
        'n_packets_with_attacker_ip': n_att_pkts,
        'n_distinct_flows': len(flow_pkts),
        'n_distinct_flows_with_attacker_ip': len(flow_attacker),
        'attacker_first_flow_rank': rank,
        'top_src_ips_by_packets': ip_counter.most_common(5),
        'top_24_prefixes': prefix_counter.most_common(5),
        'top_flows_by_packets': [
            # key = (src, sport, dst, dport, proto): the flat 5-tuple the scapy
            # extractor also uses (tuple concatenation of the two endpoints)
            {'src': k[0], 'sport': k[1], 'dst': k[2], 'dport': k[3],
             'proto': k[4], 'packets': v} for k, v in top_flows],
    }


def verify_timebase(chunks, sample, n_check=5):
    """Re-check, in-run, that a chunk's filename time really is its UTC+8 capture time.

    Reads the FIRST packet of up to n_check sampled chunks with scapy PcapReader and
    compares its epoch timestamp with the filename-derived UTC.  Informational: the
    per-chunk offset and a boolean are recorded, never asserted, so a surprise is
    reported rather than hidden.
    """
    from scapy.utils import PcapReader
    names = sample['attack'][:n_check]
    checks = []
    for name in names:
        path = os.path.join(CHUNK_DIR, name)
        r = PcapReader(path)
        first = None
        for pkt in r:
            first = float(getattr(pkt, 'time', 0.0))
            break
        r.close()
        utc = utc_from_chunk_name(name)
        if first is None:
            continue
        got = dt.datetime.utcfromtimestamp(first)
        checks.append({
            'chunk': name,
            'filename_utc': utc.strftime('%Y-%m-%d %H:%M:%S'),
            'first_packet_utc': got.strftime('%Y-%m-%d %H:%M:%S'),
            'offset_seconds': round((got - utc).total_seconds(), 3),
        })
    worst = max((abs(c['offset_seconds']) for c in checks), default=None)
    return {
        'claim': ('packet timestamps are UTC; chunk filenames encode UTC+8 (this machine); '
                  'the schedule is ADT = UTC-3'),
        'local_to_utc_offset_hours': LOCAL_TO_UTC_OFFSET_HOURS,
        'chunks_checked': checks,
        'max_abs_offset_seconds': worst,
        'all_within_5s': (worst is not None and worst <= 5.0),
    }


def span_of(chunks):
    """UTC span of the enumerated Wednesday chunk set (min start, max start)."""
    starts = [c['utc'] for c in chunks]
    return (min(starts).strftime('%Y-%m-%d %H:%M:%S'),
            max(starts).strftime('%Y-%m-%d %H:%M:%S'))


# ---------------------------------------------------------------------------
# the pipeline pass
# ---------------------------------------------------------------------------
def flow_rows_from_result(chunk_name, label_at_0, result, want_diag):
    """Per-flow records for one chunk: fusion status/alert, class, 5-tuple, _diag."""
    gnn_results = (result.get('gnn') or {}).get('results') or []
    metas = result.get('flow_metadata') or []
    rows = []
    for r in gnn_results:
        idx = r.get('flow_index', -1)
        meta = metas[idx] if (0 <= idx < len(metas)) else {}
        fusion = r.get('fusion_decision') or {}
        status = fusion.get('status')
        src = str(meta.get('src_ip', ''))
        dst = str(meta.get('dst_ip', ''))
        row = {
            'chunk': chunk_name,
            'chunk_label_at_0': label_at_0,
            'flow_index': idx,
            'status': status,
            'alert': bool(status in FUSION_ALERT_STATUSES),
            'class_id': r.get('class_id'),
            'class_name': r.get('class_name'),
            'confidence': r.get('confidence'),
            'ood': bool(r.get('ood', False)),
            'gnn_skipped': bool(r.get('gnn_skipped', False)),
            'src_ip': src, 'dst_ip': dst,
            'src_port': meta.get('src_port'), 'dst_port': meta.get('dst_port'),
            'protocol': meta.get('protocol'),
            'attacker_involved': bool(ATTACKER_IP in (src, dst)),
        }
        if want_diag:
            diag = r.get('_diag') or {}
            row['diag'] = {
                'gnn_prob_attack': diag.get('gnn_prob_attack'),
                'cluster_score': diag.get('cluster_score'),
                'is_attack': diag.get('is_attack'),
                'in_domain': diag.get('in_domain'),
                'known_proto': diag.get('known_proto'),
                'cluster_high': diag.get('cluster_high'),
                'cluster_matched': diag.get('cluster_matched'),
                'cluster_branch_structural': diag.get('cluster_branch_structural'),
            }
        else:
            row['diag'] = None
        rows.append(row)
    return rows


def run_pipeline_pass(chunk_names, chunk_lookup, max_flows, want_diag, raw_scan=True,
                      log_every=10):
    """One pipeline run per chunk, SERIALLY.  Returns (flows, per_chunk, backend_records)."""
    from nemesys_gnn4id.gnn4id.analyzer import last_feature_backend
    from nemesys_gnn4id.pipeline import TrafficPipeline

    proto_model = os.environ.get('NEMESYS_PROTO_MODEL', PROTO_MODEL_DEFAULT)
    p = TrafficPipeline(device='cpu', model_path=MODEL_PATH, proto_model_path=proto_model)
    if not p.gnn.model_loaded:
        raise RuntimeError('GNN model did not load (simulated mode would make the run '
                           'meaningless); refusing to continue')

    all_flows, per_chunk, backends, t0 = [], [], Counter(), time.time()
    for i, name in enumerate(chunk_names, 1):
        entry = chunk_lookup[name]
        t_c = time.time()
        raw = raw_ip_scan(entry['path']) if raw_scan else None
        result = p.analyze(entry['path'], max_flows=max_flows)
        rows = flow_rows_from_result(name, entry['label_at_0'], result, want_diag)
        all_flows.extend(rows)
        backend, reason = last_feature_backend()
        backends[f'{backend}|{reason}'] += 1
        record = {
            'chunk': name,
            'utc': entry['utc'].strftime('%Y-%m-%d %H:%M:%S'),
            'local_adt': entry['local'].strftime('%Y-%m-%d %H:%M:%S'),
            'chunk_label_at_0': entry['label_at_0'],
            'duration_s_estimate': entry['duration_s'],
            'n_flows': len(rows),
            'possibly_truncated_at_max_flows': len(rows) >= max_flows,
            'n_alert_flows': sum(1 for r in rows if r['alert']),
            'n_flows_with_attacker_ip': sum(1 for r in rows if r['attacker_involved']),
            'status_counts': dict(Counter(r['status'] for r in rows)),
            'class_counts': dict(Counter(str(r['class_name']) for r in rows)),
            'raw_packet_scan': raw,
            'elapsed_seconds': round(time.time() - t_c, 2),
        }
        if raw:
            # would the pipeline have seen the marker at all, given df.head(max_flows)?
            rank = raw.get('attacker_first_flow_rank')
            record['attacker_marker_visible_to_pipeline'] = (
                None if rank is None else rank <= max_flows)
        per_chunk.append(record)
        if i % log_every == 0 or i == len(chunk_names):
            print(f'[timewindow] {i}/{len(chunk_names)} chunks  '
                  f'{(time.time() - t0) / 60:.1f} min elapsed  '
                  f'{sum(c["n_flows"] for c in per_chunk)} flows', flush=True)
    return all_flows, per_chunk, {
        'env_value': os.environ.get('NEMESYS_FEATURE_BACKEND'),
        'records': dict(backends),
        'model_loaded': bool(p.gnn.model_loaded),
        'proto_model_path': proto_model,
        'model_path': MODEL_PATH,
    }


def byte_scan_for_ip(chunk_names, ips=(ATTACKER_IP,), block=1 << 22):
    """Scan the RAW BYTES of every Wednesday chunk for the 4-byte form of each IP.

    IP addresses are stored literally in the pcap packet headers, so a chunk that never
    contains these four bytes never contains a packet from or to that address.  This
    makes the absence of a marker immune to sampling and to max_flows truncation -- it
    covers the whole capture at disk speed instead of parsing it.  A hit can still be a
    PAYLOAD coincidence (the pattern is short), so hits are reported with their chunk and
    left for verification by a real parse rather than interpreted.
    """
    if isinstance(ips, str):
        ips = [ips]
    needles = {ip: bytes(int(x) for x in ip.split('.')) for ip in ips}
    per_ip = {ip: [] for ip in ips}
    n_bytes = 0
    for name in chunk_names:
        path = os.path.join(CHUNK_DIR, name)
        carry = b''
        with open(path, 'rb') as f:
            while True:
                buf = f.read(block)
                if not buf:
                    break
                n_bytes += len(buf)
                hay = carry + buf
                base = n_bytes - len(buf) - len(carry)
                for ip, needle in needles.items():
                    start = 0
                    while True:
                        i = hay.find(needle, start)
                        if i < 0:
                            break
                        per_ip[ip].append({'chunk': name, 'offset_in_chunk': base + i})
                        start = i + 1
                carry = hay[-3:]
    return {
        'chunks_scanned': len(chunk_names),
        'bytes_scanned': n_bytes,
        'results': {
            ip: {'needle_hex': needles[ip].hex(), 'n_occurrences': len(hits),
                 'occurrences': hits[:20]}
            for ip, hits in per_ip.items()},
        'interpretation': ('0 occurrences of an address proves it is in no packet of the '
                           'Wednesday capture; >0 means a possible payload coincidence that '
                           'must be verified by parsing the named chunks'),
    }


def verify_byte_hits(hit_chunks):
    """For each chunk named by a byte hit, parse IP headers and count real occurrences."""
    out = []
    from scapy.utils import PcapReader
    from scapy.layers.inet import IP
    wanted = sorted({c['ip'] for c in hit_chunks})
    for name in sorted({c['chunk'] for c in hit_chunks}):
        counts = {ip: 0 for ip in wanted}
        r = PcapReader(os.path.join(CHUNK_DIR, name))
        for pkt in r:
            if IP in pkt:
                for ip in wanted:
                    if ip in (pkt[IP].src, pkt[IP].dst):
                        counts[ip] += 1
        r.close()
        out.append({'chunk': name, 'packets_per_ip': counts})
    return out


def build_verdicts(per_margin, raw_summary):
    """Computed verdicts -- reported as measured, never tuned."""
    b0 = per_margin['0']['scheme_b_flow_level']
    a0 = per_margin['0']['scheme_a_chunk_level']
    ww = per_margin['0'].get('window_wide_flow_diagnostic', {})
    return {
        'scheme_b_viable': bool(b0['n_attack_flows'] > 0),
        'scheme_b_attack_flows': b0['n_attack_flows'],
        'marker_chunks_in_raw_packets': raw_summary.get('n_chunks_with_attacker_packets'),
        'marker_chunks_scanned': raw_summary.get('n_chunks_scanned'),
        'scheme_b_reason': (
            f'{b0["n_attack_flows"]} flow(s) satisfy "in-window AND involves {ATTACKER_IP}" '
            f'out of {b0["n_flows"]} labelled flows; the raw packet scan finds that address '
            f'in {raw_summary.get("n_chunks_with_attacker_packets")} of '
            f'{raw_summary.get("n_chunks_scanned")} sampled chunks '
            f'({raw_summary.get("total_packets_with_attacker_ip")} packets). '
            + ('Flow-level labelling by this marker yields no positive class, so recall/'
               'precision/F1 are undefined and Scheme B is NOT viable as specified.'
               if b0['n_attack_flows'] == 0 else
               'Scheme B has a positive class and its metrics are reported.')),
        'scheme_a_signals_are_real': (
            'not claimed here: inside-vs-outside alert rates are the evidence -- see '
            'scheme_a_chunk_level.per_window[*].chunk_detection_rate vs '
            'scheme_a_chunk_level.benign.chunk_false_positive_rate'),
        'scheme_a_windows_detected': {
            lab: v['chunk_detection_rate'] for lab, v in a0['per_window'].items()},
        'window_wide_diagnostic_at_default_rule': (
            ww.get('default_operating_point', {}).get('operating_point')),
        'note_on_window_wide': (
            'the window-wide numbers exist only to show what flow-level metrics look like '
            'when in-window background traffic has to be called an attack; they are a '
            'diagnostic of the labelling gap, not a competing labelling scheme'),
    }


def reanalyze(run_json, per_flow_json, out_path, per_flow_out=None):
    """Re-derive every analysis block from the single pipeline pass's per-flow dump.

    The pipeline is NOT re-run: the dump carries each flow's decision and scores, so the
    aggregates, the margin sweep and the diagnostics can be recomputed -- and new ones
    added -- from it.  Blocks the original run already computed are recomputed here and
    compared, so a divergence between the two derivations is visible instead of silent.

    When ``per_flow_out`` is given the dump is also rewritten there in COMPACT form (no
    indentation), which is the same content at a fraction of the file size; passing the
    dump's own path rewrites it in place.
    """
    with open(run_json, encoding='utf-8') as f:
        base = json.load(f)
    with open(per_flow_json, encoding='utf-8') as f:
        dump = json.load(f)
    flows, per_chunk = dump['flows'], dump['per_chunk']
    want_fpr = bool(dump['_meta'].get('ablation_dump', base['_meta'].get('ablation_dump')))
    chunks, _unparsed, _span_info = enumerate_chunks()
    chunk_lookup = {c['name']: c for c in chunks}

    per_margin = evaluate(flows, per_chunk, chunk_lookup, want_fpr=want_fpr)
    deltas = deltas_vs_margin0(per_margin)

    self_check = {
        'scheme_a_chunk_level_reproduces_runtime_block':
            _canonical(base['scheme_a_chunk_level'])
            == _canonical(per_margin['0']['scheme_a_chunk_level']),
        'scheme_b_flow_level_reproduces_runtime_block':
            _canonical(base['scheme_b_flow_level'])
            == _canonical(per_margin['0']['scheme_b_flow_level']),
        'contamination_reproduces_runtime_block':
            _canonical(base['contamination_check'])
            == _canonical(per_margin['0']['contamination']),
        'per_chunk_reproduces_runtime_block': _canonical(base['per_chunk'])
        == _canonical(per_chunk),
        'n_flows_in_dump': len(flows),
        'note': ('all True means the derivation in this file reproduces exactly what the '
                 'pipeline-pass run wrote, so the extended blocks (window-wide diagnostic) '
                 'come from the same single pass'),
        'how_equality_is_checked': (
            'via _canonical(), which maps non-finite floats to null first: with no '
            'positive class the score AUCs are NaN and bare NaN != NaN under a JSON '
            'round-trip, so a plain == on the loaded block would report a false mismatch'),
        'differing_key_paths': {
            tag: _diff_paths(base[tag], fresh)
            for tag, fresh in (
                ('scheme_a_chunk_level', per_margin['0']['scheme_a_chunk_level']),
                ('scheme_b_flow_level', per_margin['0']['scheme_b_flow_level']),
                ('contamination_check', per_margin['0']['contamination']),
                ('per_chunk', per_chunk))
            if _canonical(base[tag]) != _canonical(fresh)},
    }
    base['scheme_a_chunk_level'] = per_margin['0']['scheme_a_chunk_level']
    base['scheme_b_flow_level'] = per_margin['0']['scheme_b_flow_level']
    base['window_wide_flow_diagnostic'] = per_margin['0']['window_wide_flow_diagnostic']
    base['contamination_check'] = per_margin['0']['contamination']
    base['boundary_sensitivity'] = {'margins_minutes': list(MARGINS),
                                    'per_margin': per_margin,
                                    'deltas_vs_margin0': deltas}
    base['verdicts'] = build_verdicts(per_margin, base.get('raw_packet_marker_check', {}))
    base['_meta']['analysis_command'] = (
        f'{os.path.basename(sys.executable)} experiments/cicids2017_timewindow_eval.py '
        f'--reanalyze {os.path.relpath(run_json, REPO)} '
        f'{os.path.relpath(per_flow_json, REPO)} --out {os.path.relpath(out_path, REPO)}')
    base['_meta']['analysis_source'] = (
        'derived from the per-flow dump of the single pipeline pass (the pass itself was '
        'run once; see _meta.command)')
    base['_meta']['rederive_self_check'] = self_check

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(base), f, indent=2, ensure_ascii=False)
    if per_flow_out:
        dump['per_margin'] = per_margin
        with open(per_flow_out, 'w', encoding='utf-8') as f:
            json.dump(_json_safe(dump), f, separators=(',', ':'), ensure_ascii=False)
        print(f'[timewindow] rewrote per-flow dump compactly: {per_flow_out} '
              f'({os.path.getsize(per_flow_out) / 1e6:.1f} MB)')
    print(json.dumps(self_check, indent=2, ensure_ascii=False))
    print_report(base)
    print(f'\n[timewindow] saved {out_path}')
    return base


def raw_marker_summary(per_chunk):
    """Aggregate the raw-packet scan: is the marker in the chunk, and did we see it?"""
    have = [c for c in per_chunk if c['raw_packet_scan']]
    att = [c for c in have if c['raw_packet_scan']['n_packets_with_attacker_ip'] > 0]
    visible = [c for c in att if c.get('attacker_marker_visible_to_pipeline')]
    truncated = [c['chunk'] for c in att if c.get('attacker_marker_visible_to_pipeline') is False]
    return {
        'n_chunks_scanned': len(have),
        'n_chunks_with_attacker_packets': len(att),
        'n_chunks_where_marker_is_within_first_max_flows': len(visible),
        'n_chunks_where_marker_exists_but_is_truncated_away': len(truncated),
        'truncated_chunks': truncated[:20],
        'total_packets': sum(c['raw_packet_scan']['n_packets'] for c in have),
        'total_packets_with_attacker_ip': sum(
            c['raw_packet_scan']['n_packets_with_attacker_ip'] for c in have),
        'total_distinct_flows_raw': sum(c['raw_packet_scan']['n_distinct_flows'] for c in have),
        'total_distinct_flows_with_attacker_ip': sum(
            c['raw_packet_scan']['n_distinct_flows_with_attacker_ip'] for c in have),
        'top_src_ips_by_packets_overall': Counter(
            ip for c in have
            for ip, _n in c['raw_packet_scan']['top_src_ips_by_packets']).most_common(10),
        'top_24_prefixes_overall': Counter(
            p for c in have
            for p, _n in c['raw_packet_scan']['top_24_prefixes']).most_common(10),
        'caveat': ('top_src_ips_by_packets_overall counts how often an IP appears in a '
                   'chunk\'s top-5 list, not packet totals; per-chunk lists are in '
                   'per_chunk[].raw_packet_scan and are the authoritative detail'),
    }


# ---------------------------------------------------------------------------
# labelling + metrics (recomputed per margin from the ONE pipeline pass)
# ---------------------------------------------------------------------------
def label_flows(flows, chunk_lookup, margin_minutes):
    """(chunk_label, flow_label) for every flow at one margin.

    Scheme B: a flow is attack iff its chunk is inside a window at this margin AND the
    flow involves ATTACKER_IP.  Every other flow is benign -- including attacker-IP flows
    that fall outside every window (those are the contamination of the marker).
    """
    labels = {}
    n_changed = 0
    for name, entry in chunk_lookup.items():
        lab = window_for(entry['utc'], margin_minutes)
        labels[name] = lab
        if lab != entry['label_at_0']:
            n_changed += 1
    for r in flows:
        r['_chunk_label'] = labels[r['chunk']]
        r['_y_true'] = int(r['_chunk_label'] != 'Benign' and r['attacker_involved'])
    return labels, n_changed


def scheme_a_metrics(per_chunk, chunk_labels):
    """Chunk-level detection rate per window + benign false-positive rate."""
    by_window, by_label = {}, defaultdict(list)
    for c in per_chunk:
        by_label[chunk_labels[c['chunk']]].append(c)
    for _d, label, _s, _e, _n in day_windows():
        group = by_label.get(label, [])
        det = sum(1 for c in group if c['n_alert_flows'] > 0)
        by_window[label] = {
            'n_chunks_sampled': len(group),
            'n_chunks_detected': det,
            'chunk_detection_rate': round(det / max(len(group), 1), 6),
            'n_alert_flows': sum(c['n_alert_flows'] for c in group),
            'n_flows': sum(c['n_flows'] for c in group),
            'n_chunks_without_any_flow': sum(1 for c in group if c['n_flows'] == 0),
        }
    benign_group = by_label.get('Benign', [])
    n_fp = sum(1 for c in benign_group if c['n_alert_flows'] > 0)
    benign = {
        'n_chunks': len(benign_group),
        'n_chunks_with_alert': n_fp,
        'chunk_false_positive_rate': round(n_fp / max(len(benign_group), 1), 6),
        'n_alert_flows': sum(c['n_alert_flows'] for c in benign_group),
        'n_flows': sum(c['n_flows'] for c in benign_group),
        'n_chunks_without_any_flow': sum(1 for c in benign_group if c['n_flows'] == 0),
    }
    # chunks that changed label and are NOT one of this day's windows (e.g. margin=-1
    # can push an edge chunk into Benign, margin=+1 can do the reverse)
    n_detected = sum(v['n_chunks_detected'] for v in by_window.values())
    n_att = sum(v['n_chunks_sampled'] for v in by_window.values())
    return {
        'per_window': by_window,
        'attack_chunks_overall': {
            'n_chunks': n_att, 'n_chunks_detected': n_detected,
            'chunk_detection_rate': round(n_detected / max(n_att, 1), 6),
        },
        'benign': benign,
    }


def scheme_b_metrics(flows, want_fpr=True, y_override=None):
    """Flow-level metrics with the project's controlled-FPR convention.

    Primary row: the fusion RULE moved along the cluster_score axis (theta sweep, A4
    replay) with the operating point picked by the tie-aware benign-only rule at
    TARGET_FPR -- the repo's own convention for a rule-based variant
    (ablation_controlled_fpr.py 'A4').  Also reported: the rule at its DEFAULT operating
    point (no threshold moved), and the repo helper applied to the raw binary alert
    score, which is shown so its degeneracy on a 2-valued score is visible rather than
    hidden.

    ``y_override`` replaces the Scheme B labels (used only for the window-wide
    diagnostic, where a flow is called an attack iff its chunk is inside a window).
    """
    import numpy as np
    raf = _raf()
    if y_override is None:
        y = np.array([r['_y_true'] for r in flows], dtype=int)
        definition = ('attack iff the chunk is inside a window at this margin AND the '
                      f'flow involves {ATTACKER_IP}; otherwise benign')
    else:
        y = np.array(list(y_override), dtype=int)
        definition = ('DIAGNOSTIC labelling: attack iff the flow\'s chunk is inside a '
                      'window at this margin (every flow of the window, including benign '
                      'background traffic); this is the labelling a consumer of this '
                      'dataset is forced into without any per-flow ground truth')
    alert = np.array([bool(r['alert']) for r in flows])
    default_point = raf.binary_metrics(list(y), [int(a) for a in alert])

    out = {
        'n_flows': int(len(flows)),
        'n_attack_flows': int(y.sum()),
        'n_benign_flows': int((y == 0).sum()),
        'definition': definition,
        'default_operating_point': {
            'rule': 'fusion_decision.status in FUSION_ALERT_STATUSES (pipeline default)',
            'threshold': None,
            'operating_point': default_point,
        },
        'by_chunk_label': {},
    }
    for lab in ['Benign'] + [w[1] for w in day_windows()]:
        sel = [i for i, r in enumerate(flows) if r['_chunk_label'] == lab]
        if not sel:
            continue
        sub_y, sub_a = y[sel], alert[sel]
        out['by_chunk_label'][lab] = {
            'n_flows': len(sel),
            'n_attack_flows': int(sub_y.sum()),
            'n_benign_flows': int((sub_y == 0).sum()),
            'n_alert_flows': int(sub_a.sum()),
            'alert_rate': round(float(sub_a.mean()), 6),
            'attacker_ip_flows': int(sum(1 for i in sel if flows[i]['attacker_involved'])),
            'default_rule': raf.binary_metrics(list(sub_y), [int(a) for a in sub_a]),
        }

    if not want_fpr:
        out['controlled_fpr'] = {'available': False,
                                 'reason': 'per-flow diagnostic was off (--no-dump)'}
        return out

    diag = [r['diag'] or {} for r in flows]
    cluster_score = np.array([float(d.get('cluster_score') or 0.0) for d in diag])
    gap = np.array([np.nan if d.get('gnn_prob_attack') is None
                    else float(d['gnn_prob_attack']) for d in diag])
    is_attack = np.array([bool(d.get('is_attack')) for d in diag])
    in_domain = np.array([bool(d.get('in_domain')) for d in diag])
    known_proto = np.array([bool(d.get('known_proto')) for d in diag])
    benign = y == 0

    # ---- the fusion RULE, swept on the cluster_score axis (A4 replay) ----
    cands = np.unique(np.concatenate([
        np.unique(gap[benign][np.isfinite(gap[benign])]),   # A1 score benign values
        np.unique(cluster_score[benign]),                   # A3 score benign values
    ])) if benign.any() else np.array([])
    if len(cands) == 0:
        out['controlled_fpr'] = {'available': False,
                                 'reason': 'no benign flow carries a thresholdable score'}
        return out
    cands = np.append(cands, cands[-1] + 1e-12)

    fprs, recalls = [], []
    for theta in cands:
        high = cluster_score > theta
        pred = np.array([replay_a4_alert(is_attack[i], in_domain[i], known_proto[i],
                                         bool(high[i])) for i in range(len(flows))])
        tp, fp, fn, tn = _counts(y, pred)
        rt = _rates(tp, fp, fn, tn)
        fprs.append(rt['fpr'])
        recalls.append(rt['recall'])
    i = select_operating_point(fprs)
    pred = np.array([replay_a4_alert(is_attack[k], in_domain[k], known_proto[k],
                                     bool((cluster_score > cands[i])[k]))
                     for k in range(len(flows))])
    point = raf.binary_metrics(list(y), [int(p) for p in pred])
    assert abs(point['fpr'] - round(fprs[i], 4)) <= 1e-4, (point['fpr'], fprs[i])

    out['controlled_fpr'] = {
        'available': True,
        'target_fpr': TARGET_FPR,
        'operating_point_rule': (
            'same convention as eval_ood_method_comparison.threshold_at_target_fpr, '
            'applied to the rule-based variant exactly as ablation_controlled_fpr.py does '
            'for A4: candidates = distinct benign scores of both thresholdable scores '
            '(gnn_prob_attack, cluster_score) plus one endpoint above the largest, alert '
            'iff replayed-fusion-alert at cluster_high := cluster_score > theta, '
            'detection strict ">", operating point = candidate whose realised FPR is '
            'closest to the target, ties -> smallest theta'),
        'threshold': round(float(cands[i]), 12),
        'realised_fpr': round(float(fprs[i]), 6),
        'fpr_gap_to_target': round(abs(float(fprs[i]) - TARGET_FPR), 6),
        'operating_point': point,
        'n_theta_candidates': int(len(cands)),
        'pareto_curve': _pareto(cands, fprs, recalls),
        'self_check_rule_at_pipeline_threshold': {
            'comparison': ('replay at cluster_high := cluster_score > 0.0 vs the pipeline\'s '
                           'own cluster_high, and the replay status vs the pipeline status'),
            'n_flows': int(len(flows)),
            'n_status_mismatch': int(sum(
                1 for k, r in enumerate(flows)
                if r['status'] is not None and r['alert'] != replay_a4_alert(
                    is_attack[k], in_domain[k], known_proto[k], bool(cluster_score[k] > 0.0)))),
            'n_cluster_high_mismatch': int(sum(
                1 for k, d in enumerate(diag)
                if bool(cluster_score[k] > 0.0) != bool(d.get('cluster_high')))),
            'note': ('both counts must be 0 for the theta sweep to be an exact replay of '
                     'the pipeline rule on this corpus; a non-zero cluster_high mismatch '
                     'means some flow was flagged by the structural branch while '
                     'cluster_score == 0'),
        },
        'repo_helper_on_binary_alert_score': _binary_helper_cross_check(y, alert),
        'score_auc': _aucs(y, gap, cluster_score),
    }
    return out


def _pareto(cands, fprs, recalls):
    best = {}
    for theta, f, r in zip(cands, fprs, recalls):
        key = round(float(f), 12)
        if key not in best or r > best[key][1]:
            best[key] = (float(theta), float(r))
    return [[k, round(v[1], 6)] for k, v in sorted(best.items())]


def _binary_helper_cross_check(y, alert):
    """The repo helper run on the raw 0/1 alert score, to expose its degeneracy."""
    import numpy as np
    thr = threshold_at_target_fpr(np.asarray(alert[ y == 0], dtype=float))
    pred = np.asarray(alert, dtype=float) > thr
    tp, fp, fn, tn = _counts(y, pred)
    return {
        'helper': 'eval_ood_method_comparison.threshold_at_target_fpr on the 0/1 alert score',
        'threshold': (None if thr == float('inf') else round(float(thr), 12)),
        'operating_point': {**_round_rates(_rates(tp, fp, fn, tn)),
                            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'note': ('a 0/1 score leaves only two candidate thresholds, so the tie-aware rule '
                 'degenerates to "alert every flow" or "alert no flow"; reported for '
                 'completeness, NOT used as the Scheme B headline (the cluster_score sweep '
                 'above is the repo convention for a rule-based variant)'),
    }


def _json_safe(obj):
    """Recursively replace non-finite floats with None so the output is strict JSON.

    ``json.dump`` writes NaN/Infinity as bare literals, which strict JSON parsers reject;
    with no positive class ``roc_auc_score`` returns NaN, so this is reachable in practice.
    """
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _canonical(obj):
    """Canonical string for equality checks (NaN-safe: non-finite -> null first)."""
    return json.dumps(_json_safe(obj), sort_keys=True, ensure_ascii=False)


def _diff_paths(a, b, prefix='', limit=40):
    """Key paths where two JSON-like objects differ, so a failed self-check is legible."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f'{prefix}{k}: absent in base, present in rederived')
            elif k not in b:
                out.append(f'{prefix}{k}: present in base, absent in rederived')
            else:
                out += _diff_paths(a[k], b[k], f'{prefix}{k}.', limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f'{prefix}length: {len(a)} vs {len(b)}')
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += _diff_paths(x, y, f'{prefix}{i}.', limit)
    elif _canonical(a) != _canonical(b):
        out.append(f'{prefix[:-1]}: {a!r} vs {b!r}')
    return out[:limit]


def _aucs(y, gap, cluster_score):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    finite = np.isfinite(gap)
    one_class = ('only one class present in y (no positive flow): AUC is undefined, not 0')
    out = {}
    try:
        a = float(roc_auc_score(y, cluster_score))
        out['cluster_score'] = {
            'auc': (round(a, 4) if np.isfinite(a) else None),
            'n_flows': int(len(y)),
            'reason': (None if np.isfinite(a) else one_class),
        }
    except Exception as e:                      # one class only
        out['cluster_score'] = {'auc': None, 'reason': str(e), 'n_flows': int(len(y))}
    try:
        a = float(roc_auc_score(y[finite], gap[finite]))
        out['gnn_prob_attack'] = {
            'auc': (round(a, 4) if np.isfinite(a) else None),
            'n_flows': int(finite.sum()),
            'n_flows_without_gnn_score': int((~finite).sum()),
            'reason': (None if np.isfinite(a) else one_class),
        }
    except Exception as e:
        out['gnn_prob_attack'] = {'auc': None, 'reason': str(e),
                                  'n_flows': int(finite.sum())}
    return out


def contamination_report(flows):
    """How often ATTACKER_IP appears in flows whose chunk is OUTSIDE every window."""
    benign_chunks = sorted({r['chunk'] for r in flows if r['_chunk_label'] == 'Benign'})
    hits = [r for r in flows if r['_chunk_label'] == 'Benign' and r['attacker_involved']]
    in_window_flows = [r for r in flows if r['_chunk_label'] != 'Benign']
    per_window = {}
    for _d, label, _s, _e, _n in day_windows():
        sel = [r for r in flows if r['_chunk_label'] == label]
        per_window[label] = {
            'n_flows': len(sel),
            'n_flows_with_attacker_ip': sum(1 for r in sel if r['attacker_involved']),
            'n_chunks_with_attacker_ip': len({r['chunk'] for r in sel
                                              if r['attacker_involved']}),
        }
    return {
        'attacker_ip': ATTACKER_IP,
        'per_window_attacker_ip_flows': per_window,
        'benign_chunk_flows': len([r for r in flows if r['_chunk_label'] == 'Benign']),
        'benign_chunk_flows_with_attacker_ip': len(hits),
        'benign_chunks': len(benign_chunks),
        'benign_chunks_with_attacker_ip': len({r['chunk'] for r in hits}),
        'clean_marker': len(hits) == 0,
        'verdict': ('the attacker IP appears only inside attack windows -> usable as a flow '
                    'marker' if len(hits) == 0 else
                    'the attacker IP also carries traffic OUTSIDE the windows -> it is not a '
                    'clean attack marker; flow-level labels derived from it are contaminated'),
        'window_flows_total': len(in_window_flows),
    }


def evaluate(flows, per_chunk, chunk_lookup, margins=MARGINS, want_fpr=True):
    """Scheme A + Scheme B (+ contamination) for every margin, from the same pass."""
    sampled = [c['chunk'] for c in per_chunk]
    per_margin = {}
    for m in margins:
        labels, n_changed_all = label_flows(flows, chunk_lookup, m)
        n_changed_sample = sum(1 for n in sampled
                               if labels[n] != chunk_lookup[n]['label_at_0'])
        per_margin[str(m)] = {
            'margin_minutes': m,
            'n_chunks_changed_label_vs_margin0': n_changed_sample,
            'n_chunks_changed_label_vs_margin0_whole_day': n_changed_all,
            'scheme_a_chunk_level': scheme_a_metrics(per_chunk, labels),
            'scheme_b_flow_level': scheme_b_metrics(flows, want_fpr=want_fpr),
            'window_wide_flow_diagnostic': scheme_b_metrics(
                flows, want_fpr=want_fpr,
                y_override=[int(r['_chunk_label'] != 'Benign') for r in flows]),
            'contamination': contamination_report(flows),
        }
    # leave the flows carrying the margin-0 labels (what the per-flow dump should show)
    label_flows(flows, chunk_lookup, 0)
    return per_margin


def deltas_vs_margin0(per_margin):
    """Plain numeric deltas of the headline quantities, at every margin."""
    base = per_margin['0']
    out = {}
    for key, block in per_margin.items():
        if key == '0':
            continue
        d = {'n_chunks_changed_label_vs_margin0': block['n_chunks_changed_label_vs_margin0'],
             'n_chunks_changed_label_vs_margin0_whole_day':
                 block['n_chunks_changed_label_vs_margin0_whole_day'],
             'per_window_chunk_detection_rate': {},
             'benign_chunk_fp_rate_delta': round(
                 block['scheme_a_chunk_level']['benign']['chunk_false_positive_rate']
                 - base['scheme_a_chunk_level']['benign']['chunk_false_positive_rate'], 6)}
        for lab, v in block['scheme_a_chunk_level']['per_window'].items():
            b = base['scheme_a_chunk_level']['per_window'][lab]
            d['per_window_chunk_detection_rate'][lab] = {
                'margin': v['chunk_detection_rate'],
                'margin0': b['chunk_detection_rate'],
                'delta': round(v['chunk_detection_rate'] - b['chunk_detection_rate'], 6),
                'n_chunks_sampled': v['n_chunks_sampled'],
            }
        cb, bb = block['scheme_b_flow_level'], base['scheme_b_flow_level']
        d['scheme_b'] = {
            'n_attack_flows': {'margin': cb['n_attack_flows'], 'margin0': bb['n_attack_flows']},
            'default_recall_delta': round(
                cb['default_operating_point']['operating_point']['recall']
                - bb['default_operating_point']['operating_point']['recall'], 6),
            'default_fpr_delta': round(
                cb['default_operating_point']['operating_point']['fpr']
                - bb['default_operating_point']['operating_point']['fpr'], 6),
            'default_f1_delta': round(
                cb['default_operating_point']['operating_point']['f1']
                - bb['default_operating_point']['operating_point']['f1'], 6),
        }
        if cb.get('controlled_fpr', {}).get('available') and \
                bb.get('controlled_fpr', {}).get('available'):
            ccf, bcf = cb['controlled_fpr'], bb['controlled_fpr']
            d['scheme_b']['controlled_fpr'] = {
                'recall': {'margin': ccf['operating_point']['recall'],
                           'margin0': bcf['operating_point']['recall'],
                           'delta': round(ccf['operating_point']['recall']
                                          - bcf['operating_point']['recall'], 6)},
                'fpr': {'margin': ccf['operating_point']['fpr'],
                        'margin0': bcf['operating_point']['fpr'],
                        'delta': round(ccf['operating_point']['fpr']
                                       - bcf['operating_point']['fpr'], 6)},
                'f1': {'margin': ccf['operating_point']['f1'],
                       'margin0': bcf['operating_point']['f1'],
                       'delta': round(ccf['operating_point']['f1']
                                      - bcf['operating_point']['f1'], 6)},
            }
        out[key] = d
    return out


# ---------------------------------------------------------------------------
# inertness verification for the gated diagnostic flag
# ---------------------------------------------------------------------------
def _strip_controlled_fpr(per_margin):
    """per_margin without the controlled-FPR block (that block NEEDS the score dump)."""
    out = json.loads(json.dumps(per_margin, sort_keys=True))
    for block in out.values():
        block.get('scheme_b_flow_level', {}).pop('controlled_fpr', None)
    return json.dumps(out, sort_keys=True)


def verify_inert(path_a, path_b):
    """Compare two per-flow dumps (flag ON vs OFF) on every decision-relevant field.

    The flag can only be called inert if every DECISION and every threshold-free metric is
    identical.  The controlled-FPR block is excluded from the equality check by
    construction: it is computed FROM the dumped scores, so a run with the flag off cannot
    have it.  Its availability on each side is reported instead.
    """
    with open(path_a, encoding='utf-8') as f:
        a = json.load(f)
    with open(path_b, encoding='utf-8') as f:
        b = json.load(f)
    fa, fb = a['flows'], b['flows']
    keys = ['chunk', 'flow_index', 'status', 'alert', 'class_id', 'class_name',
            'confidence', 'ood', 'gnn_skipped', 'src_ip', 'dst_ip', 'src_port',
            'dst_port', 'protocol']
    n = min(len(fa), len(fb))
    diffs = []
    for i in range(n):
        for k in keys:
            if fa[i].get(k) != fb[i].get(k):
                diffs.append({'row': i, 'field': k, 'a': fa[i].get(k), 'b': fb[i].get(k)})
    pa, pb = a['per_chunk'], b['per_chunk']
    chunk_diffs = []
    for ca, cb in zip(pa, pb):
        for k in ('chunk', 'n_flows', 'n_alert_flows', 'n_flows_with_attacker_ip',
                  'status_counts', 'class_counts', 'raw_packet_scan'):
            if ca.get(k) != cb.get(k):
                chunk_diffs.append({'chunk': ca.get('chunk'), 'field': k,
                                    'a': ca.get(k), 'b': cb.get(k)})
    diag_a = sum(1 for r in fa if r.get('diag'))
    diag_b = sum(1 for r in fb if r.get('diag'))
    metrics_equal = _strip_controlled_fpr(a['per_margin']) == \
        _strip_controlled_fpr(b['per_margin'])
    cf = {'a_available': a['per_margin']['0']['scheme_b_flow_level']
          .get('controlled_fpr', {}).get('available'),
          'b_available': b['per_margin']['0']['scheme_b_flow_level']
          .get('controlled_fpr', {}).get('available')}
    verdict = (len(diffs) == 0 and len(chunk_diffs) == 0 and metrics_equal
               and diag_a == len(fa) and diag_b == 0
               and cf['a_available'] is True and cf['b_available'] is False)
    out = {
        'file_a': os.path.basename(path_a), 'file_a_dump_flag': a['_meta'].get('ablation_dump'),
        'file_b': os.path.basename(path_b), 'file_b_dump_flag': b['_meta'].get('ablation_dump'),
        'n_rows_compared': n,
        'flow_field_diffs': diffs[:20], 'n_flow_field_diffs': len(diffs),
        'chunk_field_diffs': chunk_diffs[:20], 'n_chunk_field_diffs': len(chunk_diffs),
        'n_rows_with_diag_in_a': diag_a, 'n_rows_with_diag_in_b': diag_b,
        'per_margin_metrics_identical_excluding_controlled_fpr': metrics_equal,
        'controlled_fpr_availability': cf,
        'verdict': (
            'INERT: with the flag ON vs OFF every per-flow decision field (status, alert, '
            'class, ood, 5-tuple), every per-chunk aggregate and every threshold-free '
            'metric is identical; the flag only adds _diag (present on '
            f'{diag_a}/{len(fa)} rows when on, {diag_b} when off). The controlled-FPR block '
            'exists only when the flag is on because it is computed from the dumped scores.'
            if verdict else 'NOT inert / comparison incomplete'),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--day', choices=sorted(DAY_CHOICES), default='Wed',
                    help='pilot day (default Wed). Wed=2017-07-05 '
                         '(DoS/Heartbleed); Fri=2017-07-07 (Botnet/PortScan/DDoS). '
                         'Wed keeps the historical output filenames unchanged.')
    ap.add_argument('--per-window', type=int, default=0,
                    help='attack chunks sampled per window (default: the module constant, '
                         '20). Used for the untruncated re-run, which needs fewer chunks '
                         'per window because each chunk now yields many more flows.')
    ap.add_argument('--n-benign', type=int, default=0,
                    help='benign chunks sampled (default: the module constant, 100)')
    ap.add_argument('--dry-run', action='store_true',
                    help='sample and label only; no pipeline, no output file')
    ap.add_argument('--limit-per-group', type=int, default=0,
                    help='smoke test: take N chunks per window and per benign group')
    ap.add_argument('--max-flows', type=int, default=DEFAULT_MAX_FLOWS)
    ap.add_argument('--seed', type=int, default=SAMPLE_SEED)
    ap.add_argument('--dump', dest='dump', action='store_true', default=True,
                    help='NEMESYS_ABLATION_DUMP=1 (default) -- needed for controlled-FPR')
    ap.add_argument('--no-dump', dest='dump', action='store_false')
    ap.add_argument('--no-raw-scan', dest='no_raw_scan', action='store_true',
                    help='skip the raw-packet attacker-IP scan inside the main loop')
    ap.add_argument('--out', default=None,
                    help='default: the day-specific path under eval_results/')
    ap.add_argument('--per-flow-out', default=None,
                    help='default: the day-specific path under eval_results/')
    ap.add_argument('--force', action='store_true',
                    help='allow overwriting an existing --out file')
    ap.add_argument('--verify-inert', nargs=2, metavar=('A_JSON', 'B_JSON'),
                    help='compare two per-flow outputs (flag on vs off) and exit')
    ap.add_argument('--byte-scan', action='store_true',
                    help='scan the raw bytes of every chunk of --day for the attacker '
                         'address (whole-capture absence check) and exit')
    ap.add_argument('--byte-scan-ips', default=None,
                    help='comma-separated IPs for --byte-scan '
                         f'(default: {" ".join(CANDIDATE_IPS)})')
    ap.add_argument('--reanalyze', nargs=2, metavar=('RUN_JSON', 'PER_FLOW_JSON'),
                    help='re-derive all analysis blocks (and the window-wide diagnostic) '
                         'from a stored per-flow dump without re-running the pipeline')
    ap.add_argument('--out-reanalyzed', default=None,
                    help='destination for --reanalyze (default: overwrite RUN_JSON); the '
                         'per-flow dump is rewritten compactly at --per-flow-out')
    args = ap.parse_args()

    # Rebind the day BEFORE any chunk enumeration or sampling. Wed keeps the historical
    # filenames so the existing committed pilot is addressable by the default path.
    global DAY_PREFIX, DAY_DATE, N_PER_WINDOW, N_BENIGN
    DAY_PREFIX, DAY_DATE = DAY_CHOICES[args.day]
    if args.per_window > 0:
        N_PER_WINDOW = args.per_window
    if args.n_benign > 0:
        N_BENIGN = args.n_benign
    if args.out is None:
        args.out = (DEFAULT_OUT if args.day == 'Wed' else
                    os.path.join(EVAL_DIR,
                                 f'cicids2017_timewindow_pilot_{args.day.lower()}.json'))
    if args.per_flow_out is None:
        args.per_flow_out = (DEFAULT_PER_FLOW_OUT if args.day == 'Wed' else
                             os.path.join(EVAL_DIR,
                                          'cicids2017_timewindow_pilot_'
                                          f'{args.day.lower()}_per_flow.json'))
    if not args.force and not args.dry_run and os.path.exists(args.out):
        raise SystemExit(f'refusing to overwrite {args.out} (pass --force)')

    if args.reanalyze:
        reanalyze(args.reanalyze[0], args.reanalyze[1],
                  args.out_reanalyzed or args.reanalyze[0],
                  per_flow_out=args.per_flow_out)
        return

    if args.byte_scan:
        names = sorted((n for n in os.listdir(CHUNK_DIR)
                        if n.startswith(f'{DAY_PREFIX}_chunk_') and n.endswith('.pcap')),
                       key=chunk_index)
        ips = [s.strip() for s in (args.byte_scan_ips or ','.join(CANDIDATE_IPS)).split(',')
               if s.strip()]
        t0 = time.time()
        scan = byte_scan_for_ip(names, ips)
        scan['elapsed_seconds'] = round(time.time() - t0, 2)
        scan['gb_scanned'] = round(scan['bytes_scanned'] / (1024 ** 3), 3)
        hits = [{'ip': ip, **h} for ip, r in scan['results'].items()
                for h in r['occurrences']]
        scan['n_total_occurrences'] = len(hits)
        if hits:
            scan['verification_by_parsing'] = verify_byte_hits(hits)
        print(json.dumps(scan, indent=2, ensure_ascii=False))
        out = os.path.join(EVAL_DIR, 'cicids2017_timewindow_marker_byte_scan.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(_json_safe({'_meta': {'command': f'{os.path.basename(sys.executable)} '
                                                       f'experiments/cicids2017_timewindow_eval.py '
                                                       f'--byte-scan' + (
                                                           f' --byte-scan-ips {args.byte_scan_ips}'
                                                           if args.byte_scan_ips else ''),
                                            'source_url': SOURCE_URL,
                                            'citation': CITATION,
                                            'attacker_ip_under_test': ATTACKER_IP,
                                            'candidate_ips_scanned': ips,
                                            'dir': os.path.relpath(CHUNK_DIR, REPO)},
                                  'scan': scan}), f, indent=2, ensure_ascii=False)
        print(f'\n[timewindow] saved {out}')
        return

    if args.verify_inert:
        verify_inert(*args.verify_inert)
        return

    os.environ.setdefault('NEMESYS_FEATURE_BACKEND', 'scapy')
    backend_forced = os.environ.get('NEMESYS_FEATURE_BACKEND')
    if args.dump:
        os.environ['NEMESYS_ABLATION_DUMP'] = '1'
    else:
        os.environ.pop('NEMESYS_ABLATION_DUMP', None)

    t_start = time.time()
    chunks, unparsed, span_info = enumerate_chunks()
    chunk_lookup = {c['name']: c for c in chunks}
    sample = select_sample(chunks, args.seed)
    overlap = span_overlap_audit(chunks, sample, span_info)
    span = span_of(chunks)

    print(f'[timewindow] {len(chunks)} {DAY_PREFIX} chunks enumerated '
          f'(unparsed: {len(unparsed)})')
    print(f'[timewindow] span {span[0]} .. {span[1]} UTC')
    print(f'[timewindow] chunk gaps: median {span_info["median_gap_s"]}s '
          f'max {span_info["max_gap_s"]}s (benign margin '
          f'{BENIGN_MARGIN_MINUTES} min -> margin_exceeds_longest_chunk='
          f'{overlap["margin_exceeds_longest_chunk"]})')
    for label, names in sample['per_window'].items():
        print(f'[timewindow]   window {label:<20} pool={sample["attack_pool_sizes"][label]:>5} '
              f'sampled={len(names)}')
    print(f'[timewindow]   benign pool={sample["benign_pool_size"]} '
          f'sampled={len(sample["benign"])} (margin {BENIGN_MARGIN_MINUTES} min)')

    sample['chunk_labels'] = {n: chunk_lookup[n]['label_at_0'] for n in
                              sample['attack'] + sample['benign']}
    sample['chunk_utc'] = {n: chunk_lookup[n]['utc'].strftime('%Y-%m-%d %H:%M:%S')
                           for n in sample['attack'] + sample['benign']}

    if args.dry_run:
        print(json.dumps({k: v for k, v in sample.items()
                          if k in ('attack_pool_sizes', 'benign_pool_size',
                                   'benign_pool_margin_minutes')},
                         indent=2, ensure_ascii=False))
        return

    run_names = list(sample['attack']) + list(sample['benign'])
    if args.limit_per_group:
        keep = []
        for _d, label, _s, _e, _n in day_windows():
            keep += sample['per_window'][label][:args.limit_per_group]
        keep += sample['benign'][:args.limit_per_group]
        run_names = keep
        print(f'[timewindow] SMOKE TEST: {len(run_names)} chunks '
              f'({args.limit_per_group} per group)')

    timebase = verify_timebase(chunks, sample)
    print(f'[timewindow] timebase check: {timebase["all_within_5s"]} '
          f'(max |offset| {timebase["max_abs_offset_seconds"]} s)')
    assert timebase['all_within_5s'], 'timebase check failed -- do not trust the labels'

    flows, per_chunk, backend = run_pipeline_pass(run_names, chunk_lookup,
                                                  args.max_flows, args.dump,
                                                  raw_scan=not args.no_raw_scan)
    per_margin = evaluate(flows, per_chunk, chunk_lookup, want_fpr=args.dump)
    deltas = deltas_vs_margin0(per_margin)
    raw_summary = raw_marker_summary(per_chunk)

    # Computed verdicts -- reported as measured, never tuned.
    verdicts = build_verdicts(per_margin, raw_summary)

    result = {
        'task': ('CICIDS2017 time-window labelling feasibility pilot -- chunk-level vs '
                 'attacker-IP flow-level labels (Wednesday 2017-07-05 only)'),
        'scope': {
            'day': 'Wednesday 2017-07-05', 'prefix': DAY_PREFIX,
            'windows_used': [w[1] for w in day_windows()],
            'n_chunks_sampled': len(run_names),
            'n_attack_chunks': len(sample['attack']) if not args.limit_per_group
            else len(run_names) - min(args.limit_per_group, len(sample['benign'])),
            'n_benign_chunks': (len(sample['benign']) if not args.limit_per_group
                                else args.limit_per_group),
            'n_flows': len(flows),
            'max_flows_per_chunk': args.max_flows,
            'smoke_test': bool(args.limit_per_group),
        },
        'scheme_a_chunk_level': per_margin['0']['scheme_a_chunk_level'],
        'scheme_b_flow_level': per_margin['0']['scheme_b_flow_level'],
        'window_wide_flow_diagnostic': per_margin['0']['window_wide_flow_diagnostic'],
        'contamination_check': per_margin['0']['contamination'],
        'raw_packet_marker_check': raw_summary,
        'verdicts': verdicts,
        'boundary_sensitivity': {'margins_minutes': list(MARGINS),
                                 'per_margin': per_margin, 'deltas_vs_margin0': deltas},
        'per_chunk': per_chunk,
        'sample': sample,
        'boundary_audit': overlap,
        'backend': backend,
        'timebase': timebase,
        'elapsed_seconds': round(time.time() - t_start, 2),
        '_meta': {
            'source_url': SOURCE_URL,
            'citation': CITATION,
            'schedule_module': 'experiments/cicids2017_schedule.py (unmodified)',
            'timebase_assertions': {
                'packet_timestamps': 'UTC',
                'chunk_filenames': 'UTC+8 (this machine local at chunking time)',
                'schedule_local_time': f'ADT = UTC{LOCAL_TO_UTC_OFFSET_HOURS:+d}',
                'wednesday_span_utc': list(span),
                'n_chunks_enumerated': len(chunks),
                'n_chunk_names_without_timestamp': len(unparsed),
                'in_run_check': timebase,
                'boundary_audit': overlap,
            },
            'sample_seed': args.seed,
            'benign_margin_minutes': BENIGN_MARGIN_MINUTES,
            'attacker_ip': ATTACKER_IP,
            'feature_backend_env': backend_forced,
            'feature_backend_actual': backend.get('records'),
            'model_path': MODEL_PATH,
            'proto_model_path': backend.get('proto_model_path'),
            'ablation_dump': bool(args.dump),
            'controlled_fpr_convention': (
                'eval_ood_method_comparison.threshold_at_target_fpr semantics (tie-aware, '
                'benign-only, strict ">", closest realised FPR to TARGET_FPR=0.20), applied '
                'to the rule-based fusion variant exactly as ablation_controlled_fpr.py '
                'does for A4 (theta sweep on cluster_score with cluster_high := '
                'cluster_score > theta)'),
            'command': ' '.join([os.path.basename(sys.executable),
                                 os.path.relpath(os.path.abspath(__file__), REPO),
                                 f'--seed {args.seed} --max-flows {args.max_flows} '
                                 f'--{"dump" if args.dump else "no-dump"}']
                                + ([f'--limit-per-group {args.limit_per_group}']
                                   if args.limit_per_group else [])),
            'env': {k: os.environ.get(k) for k in
                    ('NEMESYS_FEATURE_BACKEND', 'NEMESYS_ABLATION_DUMP',
                     'NEMESYS_PROTO_MODEL')},
            'no_pipeline_code_changed': (
                'flow 5-tuples come from the already-returned result["flow_metadata"] and '
                'the per-flow scores from the already-existing gated diagnostic '
                'NEMESYS_ABLATION_DUMP=1 (_diag); src/nemesys_gnn4id/pipeline.py is '
                'unmodified by this work'),
            'whole_capture_marker_scan': (
                'eval_results/fusion_comparison/cicids2017_timewindow_marker_byte_scan.json '
                '-- byte-level scan of ALL 2758 Wednesday chunks (12.499 GB): '
                '205.174.165.73 occurs 0 times, i.e. the address is in no packet of the '
                'capture, so its absence from the sample is NOT a sampling or truncation '
                'artefact'),
            'limitations': [
                ('flows per chunk are truncated to the FIRST max_flows flows by first-packet '
                 'order (df.head(max_flows) in the scapy extractor), so per-chunk flow '
                 'coverage is partial -- see per_chunk.possibly_truncated_at_max_flows'),
                ('scapy is the forced feature backend here (nfstream cannot open this '
                 'repo\'s non-ASCII path nor IPv6 flows); numbers are backend-specific'),
                ('the attack windows come from the dataset authors\' published schedule and '
                 'the ADT timebase is verified only through the reproduced 09:00-17:00 '
                 'capture window, not from a per-packet ground truth'),
                ('chunk labels are assigned from the chunk START time only; a chunk that '
                 'straddles a window boundary is labelled by its start (the boundary audit '
                 'reports how many sampled chunks do)'),
                ('Scheme B rests on a single published attacker address; if that address is '
                 'absent from the capture the scheme collapses, and this run measures '
                 'exactly that rather than assuming it'),
            ],
        },
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(result), f, indent=2, ensure_ascii=False)
    with open(args.per_flow_out, 'w', encoding='utf-8') as f:
        json.dump(_json_safe({'_meta': {'ablation_dump': bool(args.dump),
                                        'command': result['_meta']['command'],
                                        'n_flows': len(flows)},
                              'flows': flows, 'per_chunk': per_chunk,
                              'per_margin': per_margin}), f, indent=2,
                  ensure_ascii=False)

    print_report(result)
    print(f'\n[timewindow] saved {args.out}')
    print(f'[timewindow] saved {args.per_flow_out}')
    print(f'[timewindow] done in {result["elapsed_seconds"]}s')


def print_report(result):
    a = result['scheme_a_chunk_level']
    b = result['scheme_b_flow_level']
    c = result['contamination_check']
    print('\n' + '=' * 78)
    print('  CICIDS2017 TIME-WINDOW PILOT -- Wednesday 2017-07-05')
    print('=' * 78)
    print(f'  chunks {result["scope"]["n_chunks_sampled"]}  '
          f'flows {result["scope"]["n_flows"]}  '
          f'backend {result["_meta"]["feature_backend_actual"]}')
    print('\n  Scheme A -- chunk-level (chunk detected iff >=1 alert flow)')
    print(f'  {"window":<20} {"chunks":>7} {"detected":>9} {"rate":>8} {"flows":>7} {"alerts":>7}')
    for lab, v in a['per_window'].items():
        print(f'  {lab:<20} {v["n_chunks_sampled"]:>7} {v["n_chunks_detected"]:>9} '
              f'{v["chunk_detection_rate"]:>8.4f} {v["n_flows"]:>7} {v["n_alert_flows"]:>7}')
    print(f'  {"Benign (FP)":<20} {a["benign"]["n_chunks"]:>7} '
          f'{a["benign"]["n_chunks_with_alert"]:>9} '
          f'{a["benign"]["chunk_false_positive_rate"]:>8.4f} '
          f'{a["benign"]["n_flows"]:>7} {a["benign"]["n_alert_flows"]:>7}')
    print(f'  attack chunks overall: {a["attack_chunks_overall"]["n_chunks_detected"]}'
          f'/{a["attack_chunks_overall"]["n_chunks"]} = '
          f'{a["attack_chunks_overall"]["chunk_detection_rate"]:.4f}')

    print('\n  Scheme B -- flow-level (attack iff in-window AND involves attacker IP)')
    print(f'  flows {b["n_flows"]}  attack {b["n_attack_flows"]}  '
          f'benign {b["n_benign_flows"]}')
    d = b['default_operating_point']['operating_point']
    print(f'  default rule : rec {d["recall"]:.4f}  prec {d["precision"]:.4f}  '
          f'f1 {d["f1"]:.4f}  fpr {d["fpr"]:.4f}  (tp {d["tp"]} fp {d["fp"]} '
          f'fn {d["fn"]} tn {d["tn"]})')
    cf = b.get('controlled_fpr', {})
    if cf.get('available'):
        p = cf['operating_point']
        print(f'  @FPR~{cf["target_fpr"]:.2f}   : rec {p["recall"]:.4f}  '
              f'prec {p["precision"]:.4f}  f1 {p["f1"]:.4f}  fpr {p["fpr"]:.4f}  '
              f'(theta {cf["threshold"]}, realised {cf["realised_fpr"]:.4f})')
        sc = cf['self_check_rule_at_pipeline_threshold']
        print(f'  replay check : status mismatches {sc["n_status_mismatch"]}, '
              f'cluster_high mismatches {sc["n_cluster_high_mismatch"]} of {sc["n_flows"]}')
        print(f'  score AUC    : cluster_score {cf["score_auc"]["cluster_score"]["auc"]}, '
              f'gnn_prob_attack {cf["score_auc"]["gnn_prob_attack"]["auc"]}')
    else:
        print(f'  controlled-FPR: unavailable ({cf.get("reason")})')

    ww = result.get('window_wide_flow_diagnostic') or {}
    if ww:
        wd = ww['default_operating_point']['operating_point']
        print(f'  window-wide diagnostic (every in-window flow = attack, i.e. no per-flow '
              f'ground truth): rec {wd["recall"]:.4f}  prec {wd["precision"]:.4f}  '
              f'f1 {wd["f1"]:.4f}  fpr {wd["fpr"]:.4f}  '
              f'(attack flows {ww["n_attack_flows"]})')
        wc = ww.get('controlled_fpr', {})
        if wc.get('available'):
            wp = wc['operating_point']
            print(f'  window-wide @FPR~{wc["target_fpr"]:.2f}: rec {wp["recall"]:.4f}  '
                  f'prec {wp["precision"]:.4f}  f1 {wp["f1"]:.4f}  fpr {wp["fpr"]:.4f}')

    print('\n  Attacker-IP appearance per window (critical check)')
    for lab, v in c['per_window_attacker_ip_flows'].items():
        print(f'  {lab:<20} flows {v["n_flows"]:>6}  with {ATTACKER_IP}: '
              f'{v["n_flows_with_attacker_ip"]:>6}  in {v["n_chunks_with_attacker_ip"]} chunks')
    print(f'  benign chunks: {c["benign_chunk_flows_with_attacker_ip"]} of '
          f'{c["benign_chunk_flows"]} flows involve the IP '
          f'({c["benign_chunks_with_attacker_ip"]}/{c["benign_chunks"]} chunks) '
          f'-> clean_marker={c["clean_marker"]}')
    ba = result['boundary_audit']
    print(f'  boundary audit: benign chunks overlapping a window '
          f'{ba["benign_chunks_overlapping_any_window"]}, attack chunks overlapping two '
          f'windows {ba["attack_chunks_overlapping_two_windows"]}, '
          f'longest chunk gap {ba["chunk_gaps"]["max_gap_s"]}s')
    rm = result['raw_packet_marker_check']
    print(f'  raw-packet check: {rm["n_chunks_with_attacker_packets"]}/'
          f'{rm["n_chunks_scanned"]} chunks contain {ATTACKER_IP} '
          f'({rm["total_packets_with_attacker_ip"]} packets, '
          f'{rm["total_distinct_flows_with_attacker_ip"]} distinct flows of '
          f'{rm["total_distinct_flows_raw"]} raw); marker inside the first '
          f'{result["scope"]["max_flows_per_chunk"]} flows in '
          f'{rm["n_chunks_where_marker_is_within_first_max_flows"]} chunks, '
          f'truncated away in {rm["n_chunks_where_marker_exists_but_is_truncated_away"]}')
    shown = 0
    for c in result['per_chunk']:
        rs = c.get('raw_packet_scan')
        if not rs or c['chunk_label_at_0'] == 'Benign':
            continue
        tf = rs['top_flows_by_packets'][0] if rs['top_flows_by_packets'] else None
        print(f'    {c["chunk_label_at_0"]:<18} {c["chunk"][-28:]} '
              f'pkts={rs["n_packets"]:>6} raw_flows={rs["n_distinct_flows"]:>4} '
              f'top_src={rs["top_src_ips_by_packets"][:2]}')
        if tf:
            print(f'      heaviest flow: {tf["src"]}:{tf["sport"]} -> '
                  f'{tf["dst"]}:{tf["dport"]} proto {tf["proto"]} '
                  f'({tf["packets"]} pkts)')
        shown += 1
        if shown >= 3:
            break
    print('  verdicts: ' + json.dumps(result.get('verdicts', {}), ensure_ascii=False))

    print('\n  Boundary sensitivity (same 200 chunks, relabelled at each margin)')
    for m, blk in result['boundary_sensitivity']['per_margin'].items():
        sa = blk['scheme_a_chunk_level']
        sb = blk['scheme_b_flow_level']
        cfm = sb.get('controlled_fpr', {})
        extra = ''
        if cfm.get('available'):
            extra = (f'  f1@FPR {cfm["operating_point"]["f1"]:.4f} '
                     f'rec {cfm["operating_point"]["recall"]:.4f} '
                     f'fpr {cfm["operating_point"]["fpr"]:.4f}')
        print(f'  margin {m:>3} min: sampled chunks relabelled '
              f'{blk["n_chunks_changed_label_vs_margin0"]:>3} '
              f'(whole day {blk["n_chunks_changed_label_vs_margin0_whole_day"]:>4})  '
              f'A-det {sa["attack_chunks_overall"]["chunk_detection_rate"]:.4f}  '
              f'A-fp {sa["benign"]["chunk_false_positive_rate"]:.4f}  '
              f'B-anom {sb["n_attack_flows"]}{extra}')


if __name__ == '__main__':
    main()
