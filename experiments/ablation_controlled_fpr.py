# -*- coding: utf-8 -*-
"""Task 5: re-evaluate the fusion ablation at a MATCHED operating point.

The committed ablation table (eval_results/fusion_comparison/ablation_results.json)
reports A1..A4 at four DIFFERENT realised FPRs (0.1200 / 0.8680 / 0.6640 / 0.3640),
so the paper's claim "A4 is the only variant that combines a large recall gain over A1
with a bounded FPR" cannot be read off it: F1 at different FPRs is not comparable, and
A4 (recall .4965 / FPR .3640) is dominated by neither A1 nor A3 on both axes at once.

This script produces the comparison that actually settles it: every thresholdable
variant is moved to the operating point whose realised FPR is closest to 20% and is
then read out at THAT point, so recall is compared at equal false-alarm cost.

What is reused (not reimplemented):
  - experiments/run_ablation_fusion.py is IMPORTED. Chunk iteration, the pipeline
    setup (device/model/proto model, max_flows=50), the A1..A4 definitions and
    binary_metrics all come from it: run_ablation() below calls its main() unchanged,
    with TrafficPipeline monkeypatched to a subclass that only records the per-flow
    diagnostics. The variant arithmetic therefore cannot drift from the committed run.
  - the tie-aware matched-FPR convention (candidates just above each distinct BENIGN
    score, detection strict ">", operating point = closest realised FPR) is copied
    from experiments/eval_ood_method_comparison.py:threshold_at_target_fpr, which in
    turn documents copying it from multisignal_label_free_fusion.py.

What is new:
  - src/nemesys_gnn4id/pipeline.py emits r['_diag'] per flow when
    NEMESYS_ABLATION_DUMP=1 (default OFF; no decision logic touched).
  - the theta sweep, the A4 replay and the verification of that replay.

Variants:
  A1(theta)  alert iff gnn_prob_attack > theta      (continuous GNN score)
  A3(theta)  alert iff cluster_score  > theta       (continuous structural score)
  A4(theta)  the ACTUAL fusion rule replayed with cluster_high := (cluster_score > theta)
  A2         NOT thresholdable: it flags every OOD flow unconditionally, so it has
             exactly one operating point, which is reported separately and labelled
             in the JSON as a fixed rule.

The A4 replay is the one place a mistake would silently invalidate the result, so it
is verified two ways against the pipeline's own per-flow decisions (see
verify_replay): (1) the branch table is replayed with the pipeline's own cluster_high
boolean, which isolates the branch logic from the threshold; (2) the replay is run at
the pipeline's internal cluster threshold with cluster_high := (cluster_score > 0.0).
Both comparisons are printed and stored in the output JSON.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'src'))

import run_ablation_fusion as raf  # noqa: E402  (imported for reuse, see docstring)
from nemesys_gnn4id import pipeline as pipeline_module  # noqa: E402
from nemesys_gnn4id.gnn4id.analyzer import last_feature_backend  # noqa: E402
from nemesys_gnn4id.pipeline import (  # noqa: E402
    ABLATION_DUMP_ENV, FUSION_ALERT_STATUSES, TrafficPipeline)

TARGET_FPR = 0.20
EVAL_DIR = os.path.join('eval_results', 'fusion_comparison')
OUT_JSON = os.path.join(EVAL_DIR, 'ablation_controlled_fpr.json')
OUT_DUMP = os.path.join(EVAL_DIR, 'ablation_controlled_fpr_per_flow.json')
COMMITTED_ABLATION = os.path.join(EVAL_DIR, 'ablation_results.json')
PREFLIGHT_CHUNK = 'BenignTraffic_chunk1.pcap'

VARIANT_KEYS = {'A1': 'gnn_raw', 'A2': 'domain_routing_only',
                'A3': 'clustering_only', 'A4': 'fusion_alert'}
# the same four variants as stored per chunk by run_ablation_fusion.main()
CHUNK_METRIC_KEYS = {'A1': 'a1', 'A2': 'a2', 'A3': 'a3', 'A4': 'a4'}

# The pipeline's internal cluster threshold, expressed on cluster_score.
#
# _cluster_is_high_risk is a DISJUNCTION of two independent pieces of evidence, so it
# is not a threshold on any single raw field:
#   (i)  outlier_count > 0  (its second outlier branch, outlier_ratio > 0.15, can never
#        decide: outlier_ratio > 0.15 already implies outlier_count > 0);
#   (ii) a high-risk indicator flag AND cohesion in ('loose','moderate') -- which fires
#        with outlier_count == 0 (measured on this corpus: see
#        verification.cluster_branch_incidence).
# The per-flow cluster_score emitted by the pipeline is therefore
#   max(outlier_ratio, mean_distance if (ii) else 0.0)
# i.e. the max of two non-negative evidence strengths, which maps the method's own
# boolean onto the single strict threshold cluster_score > 0.0 exactly. The raw
# components are kept in the dump (cluster_outlier_ratio / cluster_structural_score /
# cluster_mean_distance / cluster_cohesion / cluster_indicators) so this composition is
# auditable, and an explicitly-labelled alternative is evaluated in
# result['alternate_cluster_score'].
PIPELINE_CLUSTER_THRESHOLD = 0.0

THRESHOLD_RULE = (
    'sweep theta over the ascending union of the distinct BENIGN scores of every '
    'thresholdable variant (plus one endpoint just above the largest benign score); '
    'detection uses a strict ">"; the reported operating point is the candidate whose '
    'REALISED FPR is closest to 0.20, ties resolving to the smallest theta (the '
    'tie-aware rule of eval_ood_method_comparison.py:threshold_at_target_fpr). The '
    'candidate set depends on benign scores only, never on attack labels.'
)

# ---------------------------------------------------------------------------
# per-flow diagnostic capture (the pipeline emits r['_diag'] only when the switch is on)
# ---------------------------------------------------------------------------
_RECORDS = {'backends': [], 'chunks': []}


class DiagRecordingPipeline(TrafficPipeline):
    """TrafficPipeline that records the per-flow diagnostics it just emitted.

    The override is OBSERVATIONAL: it wraps extract_features() to record which feature
    backend actually ran for this chunk, calls the parent analyze() unchanged, and
    copies r['_diag'] out of the returned result. It never alters a decision.
    """

    def analyze(self, pcap_path, *args, **kwargs):
        analyzer = self.gnn
        original_extract = analyzer.extract_features
        chunk = os.path.basename(pcap_path)

        def traced_extract(*a, **kw):
            df = original_extract(*a, **kw)
            backend, reason = last_feature_backend()
            _RECORDS['backends'].append(
                {'chunk': chunk, 'backend': backend, 'reason': reason})
            return df

        analyzer.extract_features = traced_extract
        try:
            result = super().analyze(pcap_path, *args, **kwargs)
        finally:
            analyzer.extract_features = original_extract

        flows = []
        for r in result.get('gnn', {}).get('results', []):
            diag = r.get('_diag')
            if diag is None:
                flows.append({'flow_index': r.get('flow_index'), '_diag_missing': True})
            else:
                flows.append(dict(diag, flow_index=r.get('flow_index')))
        _RECORDS['chunks'].append({'chunk': chunk, 'n_flows': len(flows), 'flows': flows})
        return result


def load_chunk_index():
    with open(os.path.join(raf.CHUNK_DIR, '_chunk_index.json'), encoding='utf-8') as f:
        return json.load(f)


def _make_subset_chunk_dir(chunk_names, index, tmp_root):
    """Temp chunk dir holding copies of `chunk_names` plus a matching _chunk_index.json."""
    subset = os.path.join(tmp_root, 'chunks')
    os.makedirs(subset, exist_ok=True)
    out_index = {}
    for name in chunk_names:
        src = os.path.join(raf.CHUNK_DIR, name)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        shutil.copyfile(src, os.path.join(subset, name))
        out_index[name] = index[name]
    with open(os.path.join(subset, '_chunk_index.json'), 'w', encoding='utf-8') as f:
        json.dump(out_index, f, ensure_ascii=False, indent=2)
    return subset


def run_ablation(chunk_names=None, dump=True, out_json=None):
    """Run run_ablation_fusion.main() verbatim, collecting the per-flow diagnostics.

    chunk_names=None runs all 70 chunks. A subset is run by pointing raf.CHUNK_DIR at a
    temp dir that holds only those chunks: main() reads CHUNK_DIR as a module global,
    so nothing else about it changes. Returns (run_results_dict, recorded_chunks).
    """
    _RECORDS['backends'].clear()
    _RECORDS['chunks'].clear()
    index = load_chunk_index()
    # run_ablation_fusion.main() imports TrafficPipeline inside the function, so the
    # class has to be swapped in its DEFINING module for the monkeypatch to take.
    original_pipeline = pipeline_module.TrafficPipeline
    original_chunk_dir = raf.CHUNK_DIR
    saved_env = {k: os.environ.get(k) for k in (ABLATION_DUMP_ENV, 'NEMESYS_ABLATION_OUT')}
    tmp_root = None
    try:
        os.environ[ABLATION_DUMP_ENV] = '1' if dump else '0'
        if chunk_names is not None or out_json is None:
            tmp_root = tempfile.mkdtemp(prefix='ablation_cfpr_', dir=REPO)
        if chunk_names is not None:
            raf.CHUNK_DIR = _make_subset_chunk_dir(chunk_names, index, tmp_root)
        run_path = out_json or os.path.join(tmp_root, 'ablation_run.json')
        os.environ['NEMESYS_ABLATION_OUT'] = run_path
        pipeline_module.TrafficPipeline = DiagRecordingPipeline
        t0 = time.time()
        raf.main()
        elapsed = time.time() - t0
        with open(run_path, encoding='utf-8') as f:
            run = json.load(f)
    finally:
        pipeline_module.TrafficPipeline = original_pipeline
        raf.CHUNK_DIR = original_chunk_dir
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)
    run['_elapsed_seconds'] = round(elapsed, 1)
    run['_chunk_subset'] = chunk_names
    run['_backend_per_chunk'] = list(_RECORDS['backends'])
    return run, [dict(c) for c in _RECORDS['chunks']]


def rows_from_flows(flows, chunk_name, label, source):
    """Flat per-flow rows for one chunk, with y_true from the chunk's label."""
    rows = []
    for flow in flows:
        row = dict(flow)
        row['chunk'] = chunk_name
        row['source'] = source
        row['attack_label'] = label
        row['attack_label_name'] = raf.ATTACK_NAMES[label]
        row['y_true'] = 0 if label == 0 else 1
        rows.append(row)
    return rows


def build_rows(chunks, index):
    """Flat per-flow rows, with y_true taken from the chunk's label in _chunk_index.json."""
    rows = []
    for chunk in chunks:
        name = chunk['chunk']
        label, source = index[name]
        rows.extend(rows_from_flows(chunk['flows'], name, int(label), source))
    return rows


# ---------------------------------------------------------------------------
# regime guard: the committed table must be reproduced, or the comparison is
# against a different extractor (nfstream and scapy give different flows)
# ---------------------------------------------------------------------------
def load_committed_ablation():
    with open(COMMITTED_ABLATION, encoding='utf-8') as f:
        return json.load(f)


def preflight_regime(chunk_name=PREFLIGHT_CHUNK, max_flows=50):
    """Run ONE chunk on the real chunk path and compare it with the committed row.

    The committed ablation table was produced with the scapy extractor (3233 flows;
    the committed run log says so), while nfstream extracts 3500 flows on this corpus
    and disagrees on EVERY chunk. A run in the wrong regime would silently produce a
    table that cannot be compared with the paper's, so the first chunk is re-run here
    and its four variant metric blocks must equal the committed ones exactly. The
    chunk path is deliberately the real (relative) one: nfstream cannot open a path
    under a non-ASCII directory, so pointing this at a copy in a temp dir would mask
    the regime it is supposed to detect.
    """
    index = load_chunk_index()
    label, source = index[chunk_name]
    saved = os.environ.get(ABLATION_DUMP_ENV)
    os.environ[ABLATION_DUMP_ENV] = '1'
    try:
        pipeline = TrafficPipeline(device='cpu', model_path=raf.MODEL_PATH,
                                   proto_model_path=raf.PROTO_MODEL)
        result = pipeline.analyze(os.path.join(raf.CHUNK_DIR, chunk_name),
                                  max_flows=max_flows)
    finally:
        if saved is None:
            os.environ.pop(ABLATION_DUMP_ENV, None)
        else:
            os.environ[ABLATION_DUMP_ENV] = saved
    flows = [r.get('_diag') for r in result.get('gnn', {}).get('results', [])]
    flows = [f for f in flows if f is not None]
    rows = rows_from_flows(flows, chunk_name, int(label), source)
    variants = pipeline_variants_from_rows(rows)
    committed = load_committed_ablation()
    committed_row = next((c for c in committed['chunk_metrics'] if c['chunk'] == chunk_name),
                         None)
    same = None
    if committed_row is not None:
        same = {tag: variants[key] == committed_row[CHUNK_METRIC_KEYS[tag]]
                for tag, key in VARIANT_KEYS.items()}
    backend, reason = last_feature_backend()
    return {
        'chunk': chunk_name,
        'n_flows': len(rows),
        'committed_n_flows': committed_row['n'] if committed_row else None,
        'feature_backend_used': backend,
        'feature_backend_reason': reason,
        'this_run': variants,
        'committed': ({CHUNK_METRIC_KEYS[tag]: committed_row[CHUNK_METRIC_KEYS[tag]]
                       for tag in VARIANT_KEYS} if committed_row else None),
        'matches_committed': same,
        'ok': bool(same) and all(same.values()),
    }


def compare_to_committed(run, committed):
    """Per-chunk comparison of this run against the committed ablation table."""
    by_name = {c['chunk']: c for c in committed['chunk_metrics']}
    compared = identical = 0
    mismatching = []
    for row in run['chunk_metrics']:
        ref = by_name.get(row['chunk'])
        if ref is None:
            continue
        compared += 1
        same = all(row[CHUNK_METRIC_KEYS[tag]] == ref[CHUNK_METRIC_KEYS[tag]]
                   for tag in VARIANT_KEYS)
        if same:
            identical += 1
        elif len(mismatching) < 10:
            mismatching.append(row['chunk'])
    return {
        'committed_file': COMMITTED_ABLATION,
        'committed_total_flows': committed['config']['total_flows'],
        'run_total_flows': run['config']['total_flows'],
        'n_chunks_compared': compared,
        'n_chunks_identical': identical,
        'first_mismatching_chunks': mismatching,
        'reproduces_committed_table': compared > 0 and identical == compared,
        'note': ('the committed table is a scapy-backend result (3233 flows on this '
                 'corpus; nfstream extracts 3500 and disagrees on every chunk), so a run '
                 'in another regime is not comparable with it and this flag is the guard.'),
    }


# ---------------------------------------------------------------------------
# the A4 replay -- must be a literal transcription of _build_fusion_decision
# ---------------------------------------------------------------------------
def replay_a4_status(is_attack, in_domain, known_proto, cluster_high):
    """Exact replica of TrafficPipeline._build_fusion_decision's branch table.

    Branch order and every condition are transcribed verbatim from
    src/nemesys_gnn4id/pipeline.py (_build_fusion_decision); only the returned dicts
    are reduced to their 'status' string, because alert-vs-not is all the ablation
    reads ('A4 = 1 if status in FUSION_ALERT_STATUSES else 0').
    """
    if is_attack and in_domain:
        return 'confirmed_gnn_attack'
    if is_attack and not in_domain and (not known_proto) and cluster_high:
        return 'suspicious_unknown_protocol_attack'
    if is_attack and not in_domain:
        return 'ood_gnn_possible_false_positive'
    if (not is_attack) and (not in_domain) and (not known_proto) and cluster_high:
        return 'possible_missed_unknown_protocol_attack'
    if (not known_proto) and (not in_domain):
        return 'unknown_protocol_monitor'
    return 'low_risk_known_protocol'


def replay_a4_alerts(is_attack, in_domain, known_proto, cluster_high):
    return replay_a4_status(is_attack, in_domain, known_proto, cluster_high) \
        in FUSION_ALERT_STATUSES


def cluster_info_from_row(row):
    """Rebuild the minimal cluster_info dict run_ablation_fusion.cluster_is_high_risk reads."""
    if not row.get('cluster_matched'):
        return None
    return {'outlier_count': row['cluster_outlier_count'],
            'size': row['cluster_size'],
            'cohesion': row['cluster_cohesion'] or '',
            'indicators': row['cluster_indicators'] or []}


def verify_replay(rows):
    """Verify the A4 replay against the pipeline's own per-flow decisions."""
    missing_status = [r for r in rows if r.get('status') is None]
    mism_branch, mism_theta = [], []
    for r in rows:
        if r.get('status') is None:
            continue
        replayed_branch = replay_a4_status(
            r['is_attack'], r['in_domain'], r['known_proto'], r['cluster_high'])
        if replayed_branch != r['status']:
            mism_branch.append({'chunk': r['chunk'], 'flow_index': r['flow_index'],
                                'pipeline': r['status'], 'replay': replayed_branch})
        # the pipeline's internal threshold on cluster_score (outlier branch)
        replayed_theta = replay_a4_status(
            r['is_attack'], r['in_domain'], r['known_proto'],
            bool(r['cluster_score'] > PIPELINE_CLUSTER_THRESHOLD))
        if replayed_theta != r['status']:
            mism_theta.append({'chunk': r['chunk'], 'flow_index': r['flow_index'],
                               'pipeline': r['status'], 'replay': replayed_theta,
                               'cluster_score': r['cluster_score'],
                               'cluster_branch_structural': r['cluster_branch_structural'],
                               'cluster_cohesion': r['cluster_cohesion'],
                               'cluster_indicators': r['cluster_indicators']})

    # which branch of _cluster_is_high_risk decided each flow
    n_outlier = sum(1 for r in rows if r['cluster_branch_outlier'])
    n_struct = sum(1 for r in rows if r['cluster_branch_structural'])
    n_struct_only = sum(1 for r in rows
                        if r['cluster_high']
                        and not (r['cluster_score'] > PIPELINE_CLUSTER_THRESHOLD))
    n_flagged = sum(1 for r in rows if r['cluster_high'])
    # does run_ablation_fusion's own replica agree with the pipeline on every flow?
    replica_diff = [r for r in rows
                    if raf.cluster_is_high_risk(cluster_info_from_row(r)) != r['cluster_high']]
    # theta_int on the score vs the pipeline's own boolean
    score_vs_high = [r for r in rows
                     if bool(r['cluster_score'] > PIPELINE_CLUSTER_THRESHOLD) != r['cluster_high']]

    def per_chunk(mismatch_list):
        counts = {}
        for r in rows:
            if r.get('status') is None:
                continue
            entry = counts.setdefault(r['chunk'], {'n_compared': 0, 'n_mismatch': 0})
            entry['n_compared'] += 1
        for m in mismatch_list:
            counts[m['chunk']]['n_mismatch'] += 1
        return [{'chunk': c, **v} for c, v in counts.items()]

    return {
        'n_flows': len(rows),
        'n_flows_without_fusion_decision': len(missing_status),
        'a4_branch_replay_vs_pipeline': {
            'comparison': ('replay_a4_status(is_attack, in_domain, known_proto, '
                           'cluster_high) vs pipeline fusion_decision.status, per flow'),
            'n_compared': len(rows) - len(missing_status),
            'n_mismatch': len(mism_branch),
            'mismatches': mism_branch[:20],
            'per_chunk': per_chunk(mism_branch),
        },
        'a4_replay_at_pipeline_cluster_threshold': {
            'comparison': ('replay_a4_status(is_attack, in_domain, known_proto, '
                           f'cluster_score > {PIPELINE_CLUSTER_THRESHOLD}) vs pipeline '
                           'fusion_decision.status, per flow'),
            'n_compared': len(rows) - len(missing_status),
            'n_mismatch': len(mism_theta),
            'mismatches': mism_theta[:20],
            'per_chunk': per_chunk(mism_theta),
        },
        'cluster_high_vs_score_at_pipeline_threshold': {
            'comparison': (f'(cluster_score > {PIPELINE_CLUSTER_THRESHOLD}) vs pipeline '
                           '_cluster_is_high_risk(cluster_info), per flow'),
            'n_mismatch': len(score_vs_high),
        },
        'cluster_branch_incidence': {
            'n_cluster_high': n_flagged,
            'n_decided_by_outlier_branch': n_outlier,
            'n_with_structural_branch_evidence': n_struct,
            'n_decided_by_structural_branch_only': n_struct_only,
            'note': ('n_decided_by_structural_branch_only is the number of flows whose '
                     'pipeline cluster_high is True while cluster_score == 0, i.e. flows '
                     'a single-theta cluster_score replay cannot reproduce. It must be 0 '
                     'for the theta_int replay to be exact.'),
        },
        'run_ablation_fusion_replica_vs_pipeline': {
            'comparison': ('run_ablation_fusion.cluster_is_high_risk(cluster_info rebuilt '
                           'from the dump) vs the pipeline\'s own _cluster_is_high_risk'),
            'n_mismatch': len(replica_diff),
        },
    }


def pipeline_variants_from_rows(rows):
    """Recompute A1..A4 from the dump using run_ablation_fusion's own definitions."""
    y_true = [r['y_true'] for r in rows]
    a1 = [0 if r['orig_class_id'] == 0 else 1 for r in rows]
    a2 = [1 if r['ood'] else a1[i] for i, r in enumerate(rows)]
    a3 = [1 if r['cluster_high'] else 0 for r in rows]
    a4 = [1 if r['status'] in FUSION_ALERT_STATUSES else 0 for r in rows]
    return {
        'gnn_raw': raf.binary_metrics(y_true, a1),
        'domain_routing_only': raf.binary_metrics(y_true, a2),
        'clustering_only': raf.binary_metrics(y_true, a3),
        'fusion_alert': raf.binary_metrics(y_true, a4),
    }


# ---------------------------------------------------------------------------
# threshold sweep
# ---------------------------------------------------------------------------
def threshold_at_target_fpr(benign_scores, target_fpr=TARGET_FPR):
    """Copied semantics of eval_ood_method_comparison.threshold_at_target_fpr."""
    b = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(b)
    if n == 0:
        return np.inf
    best_thr, best_gap = np.inf, np.inf
    for t in np.append(np.unique(b), b[-1] + 1e-12):
        gap = abs(float((b > t).sum()) / n - target_fpr)
        if gap < best_gap - 1e-12:
            best_thr, best_gap = float(t), gap
    return best_thr


def _counts(y, pred):
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    return tp, fp, fn, tn


def _rates(tp, fp, fn, tn):
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {'recall': recall, 'precision': precision, 'f1': f1, 'fpr': fpr}


def select_operating_point(candidates, fprs, recalls, target_fpr=TARGET_FPR):
    """Index of the candidate whose realised FPR is closest to target_fpr.

    candidates ascend, so this keeps the SMALLEST theta among ties -- the same
    strict-improvement rule (and therefore the same tie resolution) as
    eval_ood_method_comparison.threshold_at_target_fpr.
    """
    best_i, best_gap = 0, float('inf')
    for i, f in enumerate(fprs):
        gap = abs(float(f) - target_fpr)
        if gap < best_gap - 1e-12:
            best_i, best_gap = i, gap
    return best_i


def pareto_curve(candidates, fprs, recalls):
    """(FPR, recall) Pareto points: best recall per distinct realised FPR, ascending FPR."""
    best = {}
    for f, r in zip(fprs, recalls):
        key = round(float(f), 12)
        if key not in best or r > best[key]:
            best[key] = float(r)
    return [[k, round(best[k], 6)] for k in sorted(best)]


def sweep(rows, cluster_score_key='cluster_score'):
    """Sweep one common theta grid over A1/A3/A4 and report the matched-FPR points."""
    y = np.array([r['y_true'] for r in rows], dtype=int)
    benign = y == 0
    a1_score = np.array([np.nan if r['gnn_prob_attack'] is None else float(r['gnn_prob_attack'])
                         for r in rows], dtype=float)
    cluster_score = np.array([float(r[cluster_score_key]) for r in rows], dtype=float)
    is_attack = np.array([bool(r['is_attack']) for r in rows])
    in_domain = np.array([bool(r['in_domain']) for r in rows])
    known_proto = np.array([bool(r['known_proto']) for r in rows])

    # flows the GNN never scored (gnn_skipped OOD) can never alert in A1 at theta >= 0;
    # -inf encodes "no score" and A1's committed definition does the same (argmax of an
    # unscored flow is the default 0 = benign).
    a1_finite = np.nan_to_num(a1_score, nan=-np.inf)
    n_a1_missing = int(np.isnan(a1_score).sum())
    n_a1_missing_benign = int((np.isnan(a1_score) & benign).sum())

    cand_a1 = np.unique(a1_score[benign][np.isfinite(a1_score[benign])])
    cand_a3 = np.unique(cluster_score[benign])
    candidates = np.unique(np.concatenate([cand_a1, cand_a3]))
    if len(candidates) == 0:
        raise RuntimeError('no benign flow carries a thresholdable score; cannot sweep')
    candidates = np.append(candidates, candidates[-1] + 1e-12)

    preds = {'A1': [], 'A3': [], 'A4': []}
    for theta in candidates:
        preds['A1'].append(a1_finite > theta)
        preds['A3'].append(cluster_score > theta)
        high = cluster_score > theta
        # A4: the real fusion rule with cluster_high := (cluster_score > theta)
        preds['A4'].append(np.array([
            replay_a4_alerts(is_attack[i], in_domain[i], known_proto[i], bool(high[i]))
            for i in range(len(rows))
        ], dtype=bool))

    out = {}
    for tag in ('A1', 'A3', 'A4'):
        fprs, recalls, precisions, f1s = [], [], [], []
        for pred in preds[tag]:
            tp, fp, fn, tn = _counts(y, pred)
            rt = _rates(tp, fp, fn, tn)
            fprs.append(rt['fpr'])
            recalls.append(rt['recall'])
            precisions.append(rt['precision'])
            f1s.append(rt['f1'])
        i = select_operating_point(candidates, fprs, recalls)
        point = raf.binary_metrics(list(y), [int(p) for p in preds[tag][i]])
        # cross-check: the repo's own metric arithmetic must agree with the curve
        assert abs(point['fpr'] - round(fprs[i], 4)) <= 1e-4, (tag, point['fpr'], fprs[i])
        assert abs(point['recall'] - round(recalls[i], 4)) <= 1e-4, (tag, point['recall'], recalls[i])
        entry = {
            'threshold': round(float(candidates[i]), 12),
            'realised_fpr': round(float(fprs[i]), 6),
            'fpr_gap_to_target': round(abs(float(fprs[i]) - TARGET_FPR), 6),
            'operating_point': point,
            'curve': pareto_curve(candidates, fprs, recalls),
            'n_curve_points': len(pareto_curve(candidates, fprs, recalls)),
        }
        if tag in ('A1', 'A3'):
            # (A1/A3 have a per-flow score, so the copied repo helper applies directly; A4
            # is a rule and has no scalar score -- see _meta.limitations.)
            #
            # The helper divides the realised FPR by the number of scores it is HANDED, so
            # it must be handed one entry per BENIGN FLOW, not just the scored ones: for A1
            # 374 of the 500 benign flows are OOD-skipped and have no GNN score at all.
            # Those are encoded as -inf -- "no score", never alerts, and a legitimate left
            # endpoint of the candidate set (it alerts every scored flow). Handing the
            # helper only the finite scores silently divides its FPR by the number of
            # SCORED benign flows (126 instead of 500) and moves its operating point; that
            # bug was caught by this cross-check itself and is why the -inf encoding is
            # explicit here.
            if tag == 'A1':
                ben = np.where(np.isfinite(a1_score), a1_score, -np.inf)[benign]
            else:
                ben = np.asarray(cluster_score, dtype=float)[benign]
            ref_thr = threshold_at_target_fpr(ben)
            if tag == 'A1':
                # flows the GNN never scored carry no score at all (not a perfectly benign
                # one), so the AUC is taken over the scored flows and the count is recorded
                finite = np.isfinite(a1_score)
                entry['score_auc'] = round(float(roc_auc_score(y[finite], a1_score[finite])), 4)
                entry['score_auc_n_flows'] = int(finite.sum())
                entry['score_auc_excluded_no_score'] = int((~finite).sum())
                vec = a1_finite
            else:
                entry['score_auc'] = round(float(roc_auc_score(y, cluster_score)), 4)
                vec = cluster_score
            if ref_thr != np.inf:  # inf only when there is no benign score at all
                pred_ref = np.asarray(vec, dtype=float) > ref_thr
                tp, fp, fn, tn = _counts(y, pred_ref)
                rt = _rates(tp, fp, fn, tn)
                entry['repo_helper_cross_check'] = {
                    'helper': ('the copied eval_ood_method_comparison.threshold_at_target_fpr, '
                               'run on this variant\'s own benign scores (one entry per '
                               'benign flow, -inf where the GNN scored nothing)'),
                    'threshold': (round(float(ref_thr), 12) if np.isfinite(ref_thr) else None),
                    'threshold_is_minus_infinity': bool(not np.isfinite(ref_thr)),
                    'realised_fpr': round(rt['fpr'], 6),
                    'recall': round(rt['recall'], 6),
                    'same_realised_fpr_as_selected': abs(rt['fpr'] - fprs[i]) <= 1e-12,
                    'agrees_on_operating_point': abs(rt['fpr'] - fprs[i]) <= 1e-12
                                                 and abs(rt['recall'] - recalls[i]) <= 1e-12,
                    'note': ('the sweep grid is the union of BOTH thresholdable scores\' '
                             'distinct benign values, i.e. a superset of each variant\'s own '
                             'candidate set, so a variant can land closer to the target FPR '
                             'than this per-variant-only helper can (the union grid is the '
                             'same for every variant, so it privileges none of them). A '
                             'divergence here means the union grid reached a closer FPR. '
                             'A divergence does NOT mean the helper is wrong: both use the '
                             'same strict ">" rule and the same full-benign-pool FPR.'),
                }
        out[tag] = entry
    out['_sweep'] = {
        'n_theta_candidates': int(len(candidates)),
        'theta_min': round(float(candidates[0]), 12),
        'theta_max': round(float(candidates[-1]), 12),
        'n_theta_from_A1_score': int(len(cand_a1)),
        'n_theta_from_A3_score': int(len(cand_a3)),
        'n_flows_without_gnn_score': n_a1_missing,
        'n_benign_flows_without_gnn_score': n_a1_missing_benign,
    }
    return out


def a2_fixed_point(rows):
    """A2 is a rule, not a thresholdable detector: it flags EVERY OOD flow."""
    y_true = [r['y_true'] for r in rows]
    a1 = [0 if r['orig_class_id'] == 0 else 1 for r in rows]
    a2 = [1 if r['ood'] else a1[i] for i, r in enumerate(rows)]
    m = raf.binary_metrics(y_true, a2)
    return {
        'kind': 'fixed_rule_no_threshold',
        'rule': ('A2 = 1 if ood else (0 if _gnn_original_class_id == 0 else 1) '
                 '(experiments/run_ablation_fusion.py) -- every OOD flow is flagged '
                 'unconditionally, so there is no threshold to sweep and exactly one '
                 'operating point'),
        'threshold': None,
        'realised_fpr': m['fpr'],
        'operating_point': m,
        'curve': [[m['fpr'], m['recall']]],
    }


# ---------------------------------------------------------------------------
def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def backend_meta(records):
    counts = {}
    for entry in records:
        counts[entry['backend']] = counts.get(entry['backend'], 0) + 1
    backend, reason = last_feature_backend()
    return {
        'feature_backend_used': backend,
        'feature_backend_reason': reason,
        'feature_backend_per_chunk_counts': counts,
        'feature_backend_consistent': len(counts) <= 1,
        'feature_backend_note': ('feature_backend_used is read from '
                                 'gnn4id.analyzer.last_feature_backend() after extraction; '
                                 'the nfstream and scapy extractors are NOT equivalent '
                                 '(~49% cell agreement, different flow counts), so these '
                                 'numbers must never be mixed with a run on another backend.'),
    }


def print_report(result):
    print('\n' + '=' * 78)
    print('  ABLATION AT A MATCHED OPERATING POINT (target realised FPR = 20%)')
    print('=' * 78)
    cfg = result['pipeline_run']['config']
    print(f"  flows: {cfg['total_flows']}  chunks: {cfg['total_chunks']}  "
          f"benign: {result['_meta']['n_benign_flows']}  "
          f"attack: {result['_meta']['n_attack_flows']}")
    print(f"  feature backend: {result['_meta']['feature_backend_used']!r}")
    print()
    hdr = f'  {"variant":<34}{"theta":>12}{"FPR":>9}{"recall":>9}{"prec":>9}{"F1":>9}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for tag, name in (('A1', 'A1 GNN raw (thresholded)'),
                      ('A3', 'A3 clustering only (thresholded)'),
                      ('A4', 'A4 full fusion (rule replayed)'),
                      ('A2', 'A2 domain routing (FIXED RULE)')):
        v = result['variants'][tag]
        p = v['operating_point']
        theta = 'n/a' if v['threshold'] is None else f"{v['threshold']:.6f}"
        print(f'  {name:<34}{theta:>12}{p["fpr"]:>9.4f}{p["recall"]:>9.4f}'
              f'{p["precision"]:>9.4f}{p["f1"]:>9.4f}')
    print()
    print('  Pipeline-definition operating points (theta = pipeline thresholds, '
          'no matched FPR):')
    for tag, name in (('A1', 'A1'), ('A2', 'A2'), ('A3', 'A3'), ('A4', 'A4')):
        m = result['pipeline_definitions_operating_point'][VARIANT_KEYS[tag]]
        print(f'    {name}: FPR {m["fpr"]:.4f}  recall {m["recall"]:.4f}  F1 {m["f1"]:.4f}')
    print()
    alt = result['alternate_cluster_score']
    av = alt['variants']
    print('  Robustness check -- cluster score reduced to the outlier branch only '
          f'({alt["n_disagreements_with_pipeline_cluster_high"]} flows disagree with the '
          'pipeline rule):')
    for tag in ('A3', 'A4'):
        p = av[tag]['operating_point']
        print(f'    {tag}: theta {av[tag]["threshold"]:.6f}  FPR {p["fpr"]:.4f}  '
              f'recall {p["recall"]:.4f}  prec {p["precision"]:.4f}  F1 {p["f1"]:.4f}')
    print()
    v = result['verification']
    print(f"  replay[1] branch logic vs pipeline statuses : "
          f"{v['a4_branch_replay_vs_pipeline']['n_mismatch']} mismatches out of "
          f"{v['a4_branch_replay_vs_pipeline']['n_compared']}")
    print(f"  replay[2] at theta_int={PIPELINE_CLUSTER_THRESHOLD} on cluster_score : "
          f"{v['a4_replay_at_pipeline_cluster_threshold']['n_mismatch']} mismatches out of "
          f"{v['a4_replay_at_pipeline_cluster_threshold']['n_compared']}")
    print(f"  cluster_high == (cluster_score > {PIPELINE_CLUSTER_THRESHOLD}) : "
          f"{v['cluster_high_vs_score_at_pipeline_threshold']['n_mismatch']} mismatches")
    print(f"  structural-branch-only flows (unreproducible by the score): "
          f"{v['cluster_branch_incidence']['n_decided_by_structural_branch_only']}")
    print(f"  run_ablation_fusion's own cluster replica vs pipeline: "
          f"{v['run_ablation_fusion_replica_vs_pipeline']['n_mismatch']} mismatches")
    cc = result['_meta'].get('committed_ablation_comparison') or {}
    if cc:
        print(f"  committed table reproduced chunk-by-chunk: "
              f"{cc['n_chunks_identical']}/{cc['n_chunks_compared']} "
              f"(total flows {cc['run_total_flows']} vs committed "
              f"{cc['committed_total_flows']}) -> "
              f"{cc['reproduces_committed_table']}")
    print('=' * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chunks', type=int, default=None,
                    help='run only the first N chunks (verification runs); '
                         'default: all chunks in _chunk_index.json')
    ap.add_argument('--chunk-names', nargs='+', default=None,
                    help='run exactly these chunk names (verification runs); overrides '
                         '--chunks')
    ap.add_argument('--out', default=OUT_JSON)
    ap.add_argument('--dump-out', default=OUT_DUMP)
    ap.add_argument('--no-dump', action='store_true',
                    help='run with the diagnostic switch OFF (inertness control); no '
                         'per-flow dump is written and the sweep is skipped')
    ap.add_argument('--run-json', default=None,
                    help='also write the raw run_ablation_fusion results (config / '
                         'ablation / by_source / chunk_metrics, i.e. everything the '
                         'imported run produces) to this path, for the on/off inertness '
                         'diff -- these files must be byte-identical for the switch to '
                         'count as inert')
    ap.add_argument('--skip-preflight', action='store_true',
                    help='skip the one-chunk regime guard (it re-runs '
                         f'{PREFLIGHT_CHUNK} and compares it with the committed table)')
    ap.add_argument('--force-regime', action='store_true',
                    help='proceed even when the preflight says this run is NOT in the '
                         'committed table\'s regime (result would not be comparable)')
    args = ap.parse_args()

    index = load_chunk_index()
    chunk_names = list(index.keys())
    if args.chunk_names:
        unknown = [c for c in args.chunk_names if c not in index]
        if unknown:
            ap.error(f'unknown chunk names: {unknown}')
        chunk_names = list(args.chunk_names)
    elif args.chunks is not None:
        chunk_names = chunk_names[:args.chunks]
    subset = None if (args.chunks is None and not args.chunk_names) else chunk_names
    if subset is not None and (args.out == OUT_JSON or args.dump_out == OUT_DUMP):
        ap.error('a chunk subset must not write the full-run artifacts: pass explicit '
                 '--out and --dump-out paths (e.g. under a scratch directory)')

    preflight = None
    if not args.skip_preflight:
        print('\n[preflight] re-running one chunk to check the extractor regime against '
              f'{COMMITTED_ABLATION} ...')
        preflight = preflight_regime()
        print(f"[preflight] {preflight['chunk']}: {preflight['n_flows']} flows "
              f"(committed {preflight['committed_n_flows']}), backend "
              f"{preflight['feature_backend_used']!r}, matches committed rows: "
              f"{preflight['matches_committed']}")
        if not preflight['ok'] and not args.force_regime:
            print('!' * 78)
            print('[REGIME-MISMATCH] this run does NOT reproduce the committed ablation '
                  'table on the preflight chunk, so its numbers are not comparable with '
                  'the paper\'s table.')
            print(f"[REGIME-MISMATCH] backend={preflight['feature_backend_used']!r} "
                  f"reason={preflight['feature_backend_reason']!r}")
            print(f"[REGIME-MISMATCH] this run:   {preflight['this_run']}")
            print(f"[REGIME-MISMATCH] committed:  {preflight['committed']}")
            print('[REGIME-MISMATCH] fix: pin the committed table\'s extractor, e.g. '
                  'NEMESYS_FEATURE_BACKEND=scapy (nfstream extracts 3500 flows on this '
                  'corpus vs the committed 3233 and disagrees on every chunk), or pass '
                  '--force-regime to produce a non-comparable table anyway.')
            print('!' * 78)
            return 2

    run, chunks = run_ablation(chunk_names=subset, dump=not args.no_dump)
    if args.run_json:
        with open(args.run_json, 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in run.items() if not k.startswith('_')}, f,
                      indent=2, ensure_ascii=False)
        print(f'\n[run-json] wrote the imported run\'s own output to {args.run_json}')
    if args.no_dump:
        print(f"\n[no-dump control run] ablation block: "
              f"{json.dumps(run['ablation'], ensure_ascii=False)}")
        return 0

    rows = build_rows(chunks, index)
    assert not any(r.get('_diag_missing') for r in rows), \
        'a flow reached the result without a diagnostic record -- is the switch on?'

    verification = verify_replay(rows)
    pipeline_defs = pipeline_variants_from_rows(rows)
    agree = {tag: pipeline_defs[key] == run['ablation'][key]
             for tag, key in VARIANT_KEYS.items()}

    result = {
        '_meta': {
            'purpose': ('re-evaluate the A1..A4 fusion ablation at a MATCHED realised FPR '
                        '(target 0.20), because the committed ablation table compares '
                        'variants at four different FPRs and F1 is not comparable across '
                        'operating points'),
            'generated': datetime.datetime.now().isoformat(timespec='seconds'),
            'command': ' '.join([os.path.basename(sys.executable), *sys.argv]),
            'git_head': git_head(),
            'cwd': os.getcwd(),
            'target_fpr': TARGET_FPR,
            'threshold_rule': THRESHOLD_RULE,
            'thresholdable_variants': {
                'A1': 'alert iff gnn_prob_attack > theta, where '
                      'gnn_prob_attack = 1 - P(Benign) from the GNN4ID softmax '
                      '(analyzer._predict_with_model -> prob_benign). Flows the GNN never '
                      'scored (gnn_skipped OOD flows) have no score and never alert, '
                      'matching A1\'s committed definition (their class_id default is 0).',
                'A3': 'alert iff cluster_score > theta, where cluster_score = '
                      'outlier_count / max(size, 1) of the flow\'s matched structural '
                      'cluster (0.0 when no cluster matched; UNROUNDED, while the cluster '
                      'record itself stores a 3-decimal outlier_ratio)',
                'A4': 'the actual fusion rule replayed with '
                      'cluster_high := (cluster_score > theta) and is_attack / in_domain / '
                      'known_proto unchanged; alert iff the resulting status is in '
                      'FUSION_ALERT_STATUSES',
            },
            'fixed_rule_variant': {
                'A2': 'no threshold exists: every OOD flow is flagged unconditionally, so '
                      'A2 has exactly one operating point (reported as '
                      'variants.A2, kind=fixed_rule_no_threshold)',
            },
            'pipeline_definitions': {
                'source': 'experiments/run_ablation_fusion.py (imported, not reimplemented)',
                'A1': 'A1 = 0 if (r.get("_gnn_original_class_id", r["class_id"]) == 0) else 1',
                'A2': 'A2 = 1 if r.get("ood") else A1',
                'A3': 'A3 = 1 if cluster_is_high_risk(matched cluster) else 0',
                'A4': 'A4 = 1 if fusion_decision["status"] in FUSION_ALERT_STATUSES else 0',
            },
            'cluster_score_definition': (
                'max(outlier_count / max(size, 1), mean_distance if the matched cluster '
                'carries a high-risk indicator flag AND its cohesion is loose/moderate '
                'else 0.0); 0.0 when no cluster matched'),
            'pipeline_cluster_threshold_on_cluster_score': PIPELINE_CLUSTER_THRESHOLD,
            'pipeline_cluster_threshold_note': (
                '_cluster_is_high_risk is a disjunction of two independent pieces of '
                'evidence: (i) outlier_count > 0, and (ii) a high-risk indicator flag with '
                'cohesion in loose/moderate (which fires with outlier_count == 0). Its '
                'second outlier branch (outlier_ratio > 0.15) can never decide, because '
                'outlier_count > 0 has already returned True. cluster_score is the max of '
                'the two branch strengths, which is exactly why the pipeline\'s own '
                'boolean equals cluster_score > 0.0 -- verified per flow in '
                'verification.cluster_high_vs_score_at_pipeline_threshold.'),
            'n_chunks': run['config']['total_chunks'],
            'n_flows': run['config']['total_flows'],
            'n_benign_flows': sum(1 for r in rows if r['y_true'] == 0),
            'n_attack_flows': sum(1 for r in rows if r['y_true'] == 1),
            'flow_dump_file': args.dump_out,
            'per_flow_fields': sorted({k for r in rows for k in r}),
            'feature_backend_requested': os.environ.get('NEMESYS_FEATURE_BACKEND'),
            'feature_backend_warning': (
                'the committed ablation table is a scapy-backend result (3233 flows). '
                'This environment silently runs nfstream instead when nothing pins the '
                'backend, and nfstream extracts 3500 flows on this corpus and disagrees '
                'with the committed chunk metrics on ALL 70 chunks (measured), so a run '
                'without NEMESYS_FEATURE_BACKEND=scapy is NOT comparable with the '
                'paper\'s table. The preflight chunk and '
                'committed_ablation_comparison below are the guards for that.'),
            'limitations': [
                ('A1(theta) is a threshold on 1 - P(Benign), which is NOT the same '
                 'decision rule as the committed A1 (argmax != Benign): a flow whose '
                 'benign probability is 0.125 with a tied argmax can be alerted by A1(0.5) '
                 'while the committed A1 calls it benign. A1\'s committed operating point '
                 'is therefore reported separately under '
                 'pipeline_definitions_operating_point.'),
                ('_cluster_is_high_risk is NOT a threshold on one raw cluster field: it '
                 'has an outlier branch (outlier_count > 0) and a structural branch (a '
                 'high-risk indicator flag with cohesion loose/moderate, which fires with '
                 'outlier_count == 0 -- see '
                 'verification.cluster_branch_incidence.n_decided_by_structural_branch_'
                 'only). cluster_score therefore has to be the max of the two branch '
                 'strengths for "cluster_score > 0" to equal the pipeline\'s own boolean; '
                 'a score built from the outlier branch alone is a DIFFERENT (weaker) '
                 'detector, evaluated separately under alternate_cluster_score. The two '
                 'components live on different scales (outlier fraction vs mean centroid '
                 'distance), so theta > 0 cannot be read as a tighter outlier-ratio cutoff '
                 'on top of the unchanged pipeline rule -- it is a monotonically stricter '
                 'version of the whole cluster rule.'),
                ('the reference negatives for the threshold are the flows of the '
                 'Benign-labelled chunks (the only benign source in the corpus), the same '
                 'pool the committed ablation uses for its FPR.'),
                ('AUC (score_auc) is reported only for A1 and A3, the two variants with a '
                 'per-flow score; A4 is a rule whose threshold moves a cluster-level '
                 'boolean, so no per-flow ROC score is defined for it -- its (FPR, recall) '
                 'curve is reported instead.'),
                ('the realised FPR is always fp / (fp + tn) over ALL 500 benign flows '
                 '(the flows of the Benign-labelled chunks), the same denominator the '
                 'committed ablation uses, so the 374 benign flows the domain router sent '
                 'to the structural arm -- which the GNN never scores -- count as true '
                 'negatives for A1 rather than being dropped. A1 is therefore capped by '
                 'construction: it can never score the 1875 OOD-skipped attack flows '
                 'either, so its recall ceiling on this corpus is 858/2733 = 0.3139.'),
            ],
            **backend_meta(run['_backend_per_chunk']),
            'committed_ablation_comparison': compare_to_committed(
                run, load_committed_ablation()),
            'regime_preflight': preflight,
        },
        'pipeline_run': {'config': run['config'], 'ablation': run['ablation'],
                         'elapsed_seconds': run['_elapsed_seconds'],
                         'chunk_subset': run['_chunk_subset']},
        'pipeline_definitions_operating_point': pipeline_defs,
        'pipeline_definitions_match_run': agree,
        'verification': verification,
        'variants': {},
    }
    result['variants'].update(sweep(rows))
    result['variants']['A2'] = a2_fixed_point(rows)
    result['variants']['A1']['score'] = 'gnn_prob_attack'
    result['variants']['A3']['score'] = 'cluster_score'
    result['variants']['A4']['score'] = 'cluster_score (threshold on cluster_high)'

    # Robustness check, NOT a competing headline: the same sweep with the cluster score
    # reduced to the outlier branch alone. That score is a strictly weaker detector than
    # the pipeline's _cluster_is_high_risk (it loses every flow the structural branch
    # decided), so its A3/A4 rows are labelled and reported apart from the main table.
    alternate_disagreements = sum(
        1 for r in rows
        if bool(r['cluster_outlier_ratio'] > PIPELINE_CLUSTER_THRESHOLD) != r['cluster_high'])
    result['alternate_cluster_score'] = {
        'definition': 'outlier_count / max(size, 1) only (structural branch dropped)',
        'why_it_is_not_A4': (
            'cluster_score > 0 disagrees with the pipeline\'s own _cluster_is_high_risk on '
            f'{alternate_disagreements} of {len(rows)} flows, so a replay built on this '
            'score is a different (weaker) rule, not A4. Reported only to show how much of '
            'the result rests on the structural branch.'),
        'n_disagreements_with_pipeline_cluster_high': alternate_disagreements,
        'variants': sweep(rows, cluster_score_key='cluster_outlier_ratio'),
    }

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    dump = {'_meta': {'generated': result['_meta']['generated'],
                      'command': result['_meta']['command'],
                      'git_head': result['_meta']['git_head'],
                      'description': ('per-flow diagnostics behind the ablation variants; '
                                      'produced with NEMESYS_ABLATION_DUMP=1'),
                      'n_flows': len(rows),
                      'fields': result['_meta']['per_flow_fields']},
            'flows': rows}
    with open(args.dump_out, 'w', encoding='utf-8') as f:
        json.dump(dump, f, ensure_ascii=False, separators=(',', ':'))

    print_report(result)
    print(f"\n  saved: {args.out}")
    print(f"  saved: {args.dump_out}")
    print(f"  pipeline definitions agree with the imported ablation run: {agree}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
