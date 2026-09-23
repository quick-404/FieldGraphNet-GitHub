# -*- coding: utf-8 -*-
"""Main-line comparison: our protocol-aware system vs standard OOD detectors.

All methods are attached to the SAME base model (the GNN4ID 8-class attack
classifier), so no method is handed a stronger engine:

  GNN-raw      the model's own decision, scored as 1 - P(benign) so it can be
               thresholded like the other scores (attack iff argmax != 0)
  MSP          Hendrycks & Gimpel 2017  : 1 - max_c P(c|x)
  ODIN         Liang et al. 2018        : MSP on temperature-scaled logits (T=1000)
  Mahalanobis  Lee et al. 2018          : distance to the nearest class-conditional
                                          Gaussian fitted on IN-DOMAIN training data
                                          in the model's 16-dim penultimate embedding.
                                          Fitted in BOTH variants (known in-domain
                                          labels / the base model's own argmax) and the
                                          per-protocol headline is the stronger of the
                                          two, i.e. the baseline gets its best shot.

Feature-extractor provenance: the 82-dim features come from nfstream or from the scapy
fallback, and the two are NOT interchangeable (~49% cell agreement, 12/82 dimensions
never agree, different flow counts). `_meta.feature_backend_used` therefore records the
extractor that ACTUALLY ran (read from the analyzer after extraction), together with the
per-chunk record, and any job where more than one backend ran is flagged.

The reference file that supplies our system's rows is GATED the same way: load_reference()
withholds our-system rows (and _meta.reference_backend_mismatch /
_meta.reference_rows_withheld say so) unless the reference's recorded
`_meta.feature_backend_used` MATCHES this run's backend and its per-protocol flow counts
(n_attack / n_benign_test) agree with this run's. An absent backend in the reference
counts as a mismatch, never as a pass. Only the baselines are still reported then.

Evaluation is per OOD protocol vs a held-out benign set at a matched ~20% FPR:
every method picks its own operating point from its own benign scores with the
tie-aware "closest realised FPR" rule, and the realised FPR is reported next to
F1 (a nominal 20% FPR that silently realises something else is not a fair
comparison).

Reused, not reimplemented:
  - experiments/ood_scorers.py (commit 6735316) for msp_scores / odin_scores /
    scaled_softmax / fit_mahalanobis / mahalanobis_scores.
  - experiments/multisignal_label_free_fusion.py for the benign split convention
    (SEEDS, N_FIT/N_CALIB/N_TEST) and for `threshold_at_target_fpr`, whose small
    helper logic is COPIED here (that module imports the heavy nemesys/nfstream
    chain). `pu_two_step_detection.metrics_at_fpr` uses a plain quantile, which is
    not tie-aware, hence the copy.
  - The system's own per-protocol numbers are READ from the committed
    eval_results/fusion_comparison/complete_fusion_controlled_fpr.json key
    'Fusion-complete' (domain-routed system == structural arm on unknown
    protocols); they are not recomputed here.

Feature extraction and GNN inference run ONCE per source and are reused across
seeds; only the benign train/calib/test partition changes.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, HERE)

from nemesys_gnn4id.config import ATTACK_TYPES, LABEL_DICT
from ood_scorers import (msp_scores, odin_scores, scaled_softmax,
                         fit_mahalanobis, mahalanobis_scores)

# ---------------------------------------------------------------------------
# conventions copied from the committed reference experiments
# ---------------------------------------------------------------------------
# TARGET_FPR: identical value (0.20) in pu_two_step_detection.py and
# unknown_protocol_detection.py; inlined because importing either module pulls the
# ~30 s nfstream/scapy import chain for a single scalar.
TARGET_FPR = 0.20
# benign 3-way split convention (500 benign flows): multisignal_label_free_fusion.py
N_FIT, N_CALIB, N_TEST = 280, 70, 150
# seeds: multisignal_label_free_fusion.py / pu_two_step_detection.py
SEEDS = [42, 123, 2024, 7, 99]

IN_DOMAIN_ATTACK = 'DDoS-HTTP_Flood-.pcap'
BENIGN_SOURCE = 'BenignTraffic.pcap'
OOD_PROTOCOLS = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
                 'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']
SOURCES = [BENIGN_SOURCE, IN_DOMAIN_ATTACK] + OOD_PROTOCOLS

# 'Mahalanobis' is the HEADLINE baseline: the stronger of the two Mahalanobis variants
# per protocol (see fit_mahalanobis_variants + `mahalanobis_variant` in the JSON). Both
# variants are also reported separately so the choice is auditable.
METHODS = ['GNN-raw', 'MSP', 'ODIN', 'ODIN_T1000',
           'Mahalanobis_known', 'Mahalanobis_argmax', 'Mahalanobis']
MAHALANOBIS_HEADLINE = 'Mahalanobis'
MAHALANOBIS_VARIANTS = ('known', 'argmax')
MAHALANOBIS_VARIANT_METHOD = {'known': 'Mahalanobis_known',
                              'argmax': 'Mahalanobis_argmax'}

BENIGN_CLASS = LABEL_DICT['Benign']        # 0
IN_DOMAIN_CLASS = LABEL_DICT['DDos']       # 6 (DDoS-HTTP_Flood is a DDoS attack)
N_MODEL_CLASSES = len(ATTACK_TYPES)        # 8

EXPECTED_FLOW_DIM = 82

# ODIN temperature. Liang et al.'s CNN-scale default (1000) is nearly degenerate
# for a C=8 classifier: the MSP score ceiling is 1 - 1/8 = 0.875 and the in-domain
# score spread collapses (~1.1e-4 at T=1000 vs ~1.5e-1 at T=1), which would make the
# baseline look worse than it is and hand our system an unfair comparison. The sweep
# mirrors experiments/eval_msp_odin.py's T_SWEEP, and T is selected on IN-DOMAIN
# validation data only (never on the five test protocols); the raw CNN default stays
# reported as ODIN_T1000 for transparency.
T_SWEEP = [1.0, 10.0, 100.0, 1000.0]
ODIN_DEFAULT_T = 1000.0


def odin_key(temp):
    """Key for the per-temperature ODIN score arrays."""
    return f'ODIN_T{temp:g}'
MODEL_PATH = 'models/model.pth'

CHUNK_DIR = os.path.join(REPO, 'data', 'data', '23pcap_chunks')
CHUNK_INDEX = '_chunk_index.json'
DEFAULT_OUT = 'eval_results/fusion_comparison/ood_method_comparison.json'
REFERENCE_FILE = 'eval_results/fusion_comparison/complete_fusion_controlled_fpr.json'
REFERENCE_KEY = 'Fusion-complete'


# ---------------------------------------------------------------------------
# copied helpers (small, provenance documented) — see module docstring
# ---------------------------------------------------------------------------
def split_benign(n_benign, seed):
    """3-way benign split: fit / calibration / test. Returns index arrays.

    Copied from experiments/multisignal_label_free_fusion.py:split_benign (same
    RNG, same order, same N_FIT/N_CALIB/N_TEST), so the seed-42 test split is
    byte-identical to the partition behind complete_fusion_controlled_fpr.json.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_benign)
    fit_idx = idx[:N_FIT]
    calib_idx = idx[N_FIT:N_FIT + N_CALIB]
    test_idx = idx[N_FIT + N_CALIB:N_FIT + N_CALIB + N_TEST]
    return fit_idx, calib_idx, test_idx


def threshold_at_target_fpr(benign_scores, target_fpr=TARGET_FPR):
    """Benign-only threshold whose REALISED FPR is closest to target_fpr.

    Copied verbatim in semantics from
    experiments/multisignal_label_free_fusion.py:threshold_at_target_fpr: candidate
    thresholds are just above each distinct benign score, ties are broken towards
    the closest realised FPR, and detection uses a strict '>'. Label-free: it never
    looks at attack labels. Needed because scores can tie heavily, in which case a
    plain quantile + '>' quietly operates away from the nominal 20% FPR.
    """
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


def metrics_matched_fpr(scores, y, thr_scores=None):
    """auc/recall/precision/f1 + REALISED fpr at the benign-only matched-FPR point.

    By default benign_scores are the method's own scores on the held-out benign TEST
    flows (y == 0) — the same convention as multisignal_label_free_fusion.py and
    complete_fusion_controlled_fpr.py, which both pass s[y == 0].

    ``thr_scores`` overrides WHERE the threshold is chosen, independently of where it is
    MEASURED. That separation is the point of the independent-calibration variant: when
    the threshold is selected on the same benign flows whose realised FPR is then
    reported, the reported FPR is close to the 20% target by construction rather than by
    merit. Passing the calibration split here selects the operating point on data the
    reported metrics never see. ``thr_scores=None`` reproduces the historical behaviour
    exactly.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    benign_scores = (scores[y == 0] if thr_scores is None
                     else np.asarray(thr_scores, dtype=float))
    auc = float(roc_auc_score(y, scores))
    thr = threshold_at_target_fpr(benign_scores)
    pred = (scores > thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    fpr = fp / max(int((y == 0).sum()), 1)
    return {'auc': round(auc, 4), 'recall': round(rec, 4), 'precision': round(prec, 4),
            'f1': round(f1, 4), 'fpr': round(fpr, 4), 'threshold': round(float(thr), 6)}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _chunk_sort_key(name):
    stem = name[:-5] if name.endswith('.pcap') else name
    tail = stem.rsplit('chunk', 1)[-1]
    return int(tail) if tail.isdigit() else 0


def chunks_by_source():
    """({source: [chunk names in numeric order]}, {source: [chunk labels]}).

    Both come from _chunk_index.json. The label was previously discarded here; it is now
    returned alongside the chunk names so `assert_source_class_pure()` can check the
    hardcoded in-domain class indices (BENIGN_CLASS / IN_DOMAIN_CLASS) against the index
    instead of trusting them. Labels stay in the same order as the chunk names.
    """
    with open(os.path.join(CHUNK_DIR, CHUNK_INDEX), encoding='utf-8') as f:
        raw = json.load(f)
    chunks_by_src, labels_by_src = {}, {}
    for chunk_name, (label, source) in raw.items():
        chunks_by_src.setdefault(source, []).append(chunk_name)
        labels_by_src.setdefault(source, []).append(label)
    for source in chunks_by_src:
        order = sorted(range(len(chunks_by_src[source])),
                       key=lambda i: _chunk_sort_key(chunks_by_src[source][i]))
        chunks_by_src[source] = [chunks_by_src[source][i] for i in order]
        labels_by_src[source] = [labels_by_src[source][i] for i in order]
    return chunks_by_src, labels_by_src


def assert_source_class_pure(source, labels, expected_class):
    """Assert every chunk of `source` carries the one hardcoded in-domain class.

    Two blocks are built from hardcoded class indices rather than from labels:
    the benign-fit block (BENIGN_SOURCE chunks, used to fit Mahalanobis and to fit/select
    the ODIN temperature) assumes Benign(0), and the in-domain-attack block
    (IN_DOMAIN_ATTACK chunks) assumes DDos(6). If the chunk index were ever re-cut so
    that another class's chunks land under either source, every in-domain fit and the
    ODIN temperature selection would be silently mislabelled -- a wrong number, not a
    crash. The labels are right there in the index, so the assumption is checked.

    Raises ValueError if any label differs, or if the source has no chunks at all (an
    unverifiable block is not an acceptable block; the run would then die on empty
    extraction anyway, but with a less specific message).
    """
    if not labels:
        raise ValueError(
            f'{source}: no chunks in {CHUNK_INDEX}; cannot verify that its block is '
            f'class-pure {ATTACK_TYPES[expected_class]["name_en"]}({expected_class})')
    unexpected = sorted({int(v) for v in labels if int(v) != int(expected_class)})
    if unexpected:
        names = ', '.join(f'{ATTACK_TYPES.get(v, {}).get("name_en", "unknown")}({v})'
                          for v in unexpected)
        raise ValueError(
            f'{source}: expected every chunk to be labelled '
            f'{ATTACK_TYPES[expected_class]["name_en"]}({expected_class}), but found '
            f'{names} among {len(labels)} chunks. The in-domain blocks are built from '
            f'hardcoded class indices, so this would silently mislabel the fit data; '
            f'refusing to continue.')


def extract_source(analyzer, chunk_names, max_flows, extraction_log=None):
    """Logits (n,8) and 16-dim penultimate embeddings (n,16) for every flow.

    Uses the verified per-flow sequence: one graph per flow, one forward pass
    through the 3-arg model call, embedding captured by a forward hook on
    graph_prediction_1.

    `extraction_log` (optional dict) records, per chunk, which feature backend
    actually ran and which chunks produced no flows / no graphs, so nothing is
    dropped without a trace in the JSON (previously a chunk that yielded nothing was
    only mentioned on stdout).
    """
    logits_all, embed_all = [], []
    # imported here (not at module level) so the heavy analyzer/nfstream chain stays
    # deferred to main(), as the module docstring requires
    last_feature_backend = None
    if extraction_log is not None:
        from nemesys_gnn4id.gnn4id.analyzer import last_feature_backend as _lfb
        last_feature_backend = _lfb
    for chunk_name in chunk_names:
        path = os.path.join(CHUNK_DIR, chunk_name)
        df = analyzer.extract_features(path, max_flows=max_flows)
        if extraction_log is not None:
            # read AFTER the call: records the backend that actually produced these
            # features (not the one requested)
            backend, reason = last_feature_backend()
            extraction_log.setdefault('backends', []).append(
                {'chunk': chunk_name, 'backend': backend, 'reason': reason})
        if df is None or df.empty or 'flow_features' not in df.columns:
            print(f'    {chunk_name}: no flows')
            if extraction_log is not None:
                extraction_log.setdefault('chunks_without_flows', []).append(chunk_name)
            continue
        graphs = analyzer.build_graphs(df, expected_flow_dim=EXPECTED_FLOW_DIM)
        if not graphs and extraction_log is not None:
            extraction_log.setdefault('chunks_without_graphs', []).append(chunk_name)
        for graph in graphs:
            captured = {}
            hook = analyzer.model.graph_prediction_1.register_forward_hook(
                lambda m, inp, out: captured.__setitem__(
                    'embed', out.detach().cpu().numpy()))
            try:
                batch = Batch.from_data_list([graph]).to(analyzer.device)
                with torch.no_grad():
                    logits = analyzer.model(batch.x_dict, batch.edge_index_dict, batch)
            finally:
                hook.remove()
            logits_all.append(np.asarray(logits.detach().cpu().numpy())[0])
            embed_all.append(np.asarray(captured['embed'])[0])
        print(f'    {chunk_name}: {len(graphs)} flows')
    if not logits_all:
        return np.zeros((0, N_MODEL_CLASSES)), np.zeros((0, 16))
    return np.array(logits_all, dtype=np.float64), np.array(embed_all, dtype=np.float64)


# ---------------------------------------------------------------------------
# Mahalanobis on in-domain data only
# ---------------------------------------------------------------------------
def fit_mahalanobis_variant(X_benign_fit, X_indomain_attack, labels, label_source):
    """One Mahalanobis variant: pooled-covariance class-conditional Gaussians.

    Both variants fit on the SAME in-domain flow set (the benign fit split plus every
    extracted in-domain DDoS flow) and differ ONLY in `labels`:
      'known'  -> the KNOWN in-domain labels (Benign / DDos)
      'argmax' -> the base model's own argmax prediction (needs no labels at all)

    `fit_mahalanobis` requires CONTIGUOUS integer labels in [0, n_classes) and raises
    otherwise (commit 0c3bb6d). In-domain data covers only a subset of the model's 8
    classes, so the labels present are remapped to a contiguous 0..K-1 fit index here
    -- explicitly, once per seed, and recorded in the returned info -- rather than
    handing `fit_mahalanobis` a sparse label range (it would refuse, correctly) or
    silently pretending an 8-class fit happened. A label outside [0, N_MODEL_CLASSES)
    is a programming error and is rejected loudly.
    """
    if len(X_benign_fit) == 0 or len(X_indomain_attack) == 0:
        raise ValueError(
            'Mahalanobis fitting needs in-domain flows: got '
            f'{len(X_benign_fit)} benign fit and {len(X_indomain_attack)} '
            'in-domain attack embeddings')
    X = np.vstack([X_benign_fit, X_indomain_attack])
    y = np.asarray(labels)
    if y.shape != (len(X),):
        raise ValueError(f'got {y.shape} labels for {len(X)} fit flows')
    present = sorted(int(v) for v in np.unique(y))
    if present[0] < 0 or present[-1] >= N_MODEL_CLASSES:
        raise ValueError(
            f'labels must lie in [0, {N_MODEL_CLASSES}) (model class indices); '
            f'got [{present[0]}, {present[-1]}]')
    remap = {c: i for i, c in enumerate(present)}
    y_fit = np.array([remap[int(v)] for v in y], dtype=int)
    mu, prec = fit_mahalanobis(X, y_fit, len(present))
    info = {
        'n_fit_flows': int(len(X)),
        'n_benign_fit': int(len(X_benign_fit)),
        'n_indomain_attack': int(len(X_indomain_attack)),
        'label_source': label_source,
        'n_classes_used': len(present),
        'classes_used': present,
        'classes_used_names': [ATTACK_TYPES[c]['name_en'] for c in present],
        'label_remap_to_fit_index': {str(c): i for i, c in enumerate(present)},
        'contiguous_remap_needed': present != list(range(len(present))),
    }
    return mu, prec, info


def fit_mahalanobis_known(X_benign_fit, X_indomain_attack):
    """Variant 'known': the textbook Lee et al. fit on the KNOWN in-domain labels."""
    labels = np.array([BENIGN_CLASS] * len(X_benign_fit)
                      + [IN_DOMAIN_CLASS] * len(X_indomain_attack))
    return fit_mahalanobis_variant(
        X_benign_fit, X_indomain_attack, labels,
        'known in-domain labels (benign-fit -> '
        f'{ATTACK_TYPES[BENIGN_CLASS]["name_en"]}({BENIGN_CLASS}), in-domain DDoS -> '
        f'{ATTACK_TYPES[IN_DOMAIN_CLASS]["name_en"]}({IN_DOMAIN_CLASS}))')


def fit_mahalanobis_argmax(X_benign_fit, X_indomain_attack, logits_benign_fit,
                           logits_indomain_attack):
    """Variant 'argmax': same flows, labels = the base model's own prediction."""
    labels = np.concatenate([np.asarray(logits_benign_fit).argmax(axis=1),
                             np.asarray(logits_indomain_attack).argmax(axis=1)])
    return fit_mahalanobis_variant(
        X_benign_fit, X_indomain_attack, labels,
        'base model argmax prediction on the same in-domain fit flows (label-free)')


def fit_mahalanobis_variants(X_benign_fit, X_indomain_attack, logits_benign_fit,
                             logits_indomain_attack):
    """Both variants on the same flows -> {variant: (mu, precision, info)}."""
    return {
        'known': fit_mahalanobis_known(X_benign_fit, X_indomain_attack),
        'argmax': fit_mahalanobis_argmax(X_benign_fit, X_indomain_attack,
                                         logits_benign_fit, logits_indomain_attack),
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def aggregate(per_seed):
    """mean/std over seeds, per protocol, per method."""
    out = {}
    for proto in OOD_PROTOCOLS:
        out[proto] = {}
        for method in METHODS:
            agg = {}
            for key in ('auc', 'recall', 'precision', 'f1', 'fpr'):
                vals = [per_seed[s]['protocols'][proto]['methods'][method][key]
                        for s in per_seed]
                agg[f'{key}_mean'] = round(float(np.mean(vals)), 4)
                agg[f'{key}_std'] = round(float(np.std(vals)), 4)
            out[proto][method] = agg
    return out


def _flowcount_diff_summary(diff):
    """Compact 'n_attack: reference=.. run=..' rendering of one protocol's diff."""
    return ', '.join(f'{field}: reference={v["reference"]!r} run={v["run"]!r}'
                     for field, v in sorted(diff.items()))


def load_reference(ref_file=None, run_backend=None, run_flow_counts=None):
    """Our system's committed per-protocol numbers (read, never recomputed).

    ref_file defaults to REFERENCE_FILE. Pass --reference to select it.

    `run_backend` is the feature backend THIS run actually used, read from
    last_feature_backend() AFTER extraction and threaded in from main() (reading it in
    here would be wrong: this function can be called before extraction, when the record
    is still empty/None). `run_flow_counts` is
    {protocol: {'n_attack': int, 'n_benign_test': int}} for this run; when it is None the
    flow-count cross-check is skipped (there is nothing to compare against).

    WHY THIS VERIFIES RATHER THAN JUST LOADS: the table in the caller juxtaposes our
    system's rows, READ from this file, with baselines COMPUTED by the current run. Both
    must come from the same feature backend -- the nfstream and scapy extractors produce
    non-equivalent 82-dim features (~49% cell agreement, 12/82 dimensions never agree)
    and do not even extract the same number of flows (3233 vs 3500 on this corpus). Two
    independent checks are applied:

      (1) `_meta.feature_backend_used` must be PRESENT and equal `run_backend`. An absent
          backend is treated as a MISMATCH, not a pass: a file with no backend field
          cannot be trusted to match.
      (2) the reference's own per-protocol flow counts (n_attack / n_benign_test, stored
          by the producer and previously read and thrown away) must equal this run's.
          This catches a backend change -- or a re-cut corpus -- even when (1) is
          missing or wrong.

    On either failure the our-system rows are WITHHELD (`per_protocol` empty, the tables
    print baselines only) and the result carries reference_backend_mismatch,
    reference_flowcount_mismatch, both backend values and a human-readable reason; a loud
    banner is printed. This deliberately does NOT raise: the baseline half of the
    comparison is still valid and useful, only our-system rows must be withheld. main()
    propagates the flags into the output JSON's _meta and prints them, so nobody can read
    the table as if our-system rows were comparable.

    On success the returned dict is exactly the historical one plus the verification
    record (reference_backend_mismatch=False, both backend values, no flowcount mismatch).
    """
    ref_file = ref_file or REFERENCE_FILE
    path = os.path.join(REPO, ref_file)
    if not os.path.exists(path):
        print(f'[WARNING] reference file missing, our-system rows will be empty: {path}')
        return None
    with open(path, encoding='utf-8') as f:
        ref = json.load(f)
    per_proto = {}
    missing = []
    for proto in OOD_PROTOCOLS:
        entry = ref.get(proto, {})
        if REFERENCE_KEY in entry:
            per_proto[proto] = dict(entry[REFERENCE_KEY])
        else:
            missing.append(proto)
    if missing:
        print(f'[WARNING] our system has no {REFERENCE_KEY} entry for {missing} in '
              f'{ref_file}; those rows are left out of the reference table '
              '(and _meta.our_system_provenance) rather than guessed')

    # -- check (1): the reference must name the backend it was produced with --------
    ref_meta = ref.get('_meta') or {}
    ref_backend = ref_meta.get('feature_backend_used')
    ref_backend_reason = ref_meta.get('feature_backend_reason')
    if ref_backend is None:
        backend_problem = (
            f'the reference records NO feature backend ({ref_file} has no '
            f'_meta.feature_backend_used); an unlabelled reference cannot be trusted to '
            f'match this run\'s backend {run_backend!r}')
    elif run_backend is None:
        backend_problem = (
            'this run recorded NO feature backend (last_feature_backend() is empty), so '
            f'a match against the reference\'s {ref_backend!r} cannot be established')
    elif ref_backend != run_backend:
        backend_problem = (
            f'the reference was produced with feature backend {ref_backend!r} but this '
            f'run used {run_backend!r}')
    else:
        backend_problem = None

    # -- check (2): the reference's own flow counts must corroborate the backend -----
    flowcount_mismatch = {}
    if run_flow_counts is not None:
        for proto in OOD_PROTOCOLS:
            ref_entry = ref.get(proto)
            if not isinstance(ref_entry, dict):
                continue  # no reference entry at all: already reported by `missing`
            run_entry = run_flow_counts.get(proto) or {}
            diff = {}
            for field in ('n_attack', 'n_benign_test'):
                ref_val, run_val = ref_entry.get(field), run_entry.get(field)
                if ref_val is None or run_val is None:
                    diff[field] = {'reference': ref_val, 'run': run_val}
                elif int(ref_val) != int(run_val):
                    diff[field] = {'reference': int(ref_val), 'run': int(run_val)}
            if diff:
                flowcount_mismatch[proto] = diff

    backend_mismatch = backend_problem is not None
    if backend_mismatch:
        print('!' * 78)
        print('[REFERENCE-BACKEND-MISMATCH] our-system rows are WITHHELD.')
        print(f'[REFERENCE-BACKEND-MISMATCH] {backend_problem}')
        print(f'[REFERENCE-BACKEND-MISMATCH] reference={ref_file} '
              f'backend={ref_backend!r} (recorded reason={ref_backend_reason!r}) | '
              f'this run backend={run_backend!r}')
        print('[REFERENCE-BACKEND-MISMATCH] nfstream and scapy are NOT interchangeable '
              '(~49% cell agreement, 12/82 dimensions never agree, different flow '
              'counts), so our system\'s per-protocol rows cannot be juxtaposed with '
              'this run\'s baselines.')
        print('[REFERENCE-BACKEND-MISMATCH] fix: re-produce the reference with the same '
              'NEMESYS_FEATURE_BACKEND as this run, or re-run on the reference\'s '
              'backend.')
        print('!' * 78)
    if flowcount_mismatch:
        print('!' * 78)
        print('[REFERENCE-FLOWCOUNT-MISMATCH] our-system rows are WITHHELD: the '
              f'reference and this run disagree on the extracted flows for '
              f'{sorted(flowcount_mismatch)}')
        for proto in sorted(flowcount_mismatch):
            print(f'[REFERENCE-FLOWCOUNT-MISMATCH]   {proto}: '
                  f'{_flowcount_diff_summary(flowcount_mismatch[proto])}')
        print('[REFERENCE-FLOWCOUNT-MISMATCH] the two sources do not describe the same '
              'flows -- a different feature backend (they extract different flow counts) '
              'or a re-cut corpus/chunk index -- so our-system rows are not comparable '
              'even though the recorded backends agree.')
        print('!' * 78)

    if backend_mismatch or flowcount_mismatch:
        reason_parts = []
        if backend_mismatch:
            reason_parts.append('feature-backend mismatch: ' + backend_problem)
        if flowcount_mismatch:
            reason_parts.append(
                'flow-count mismatch: the reference and this run extracted different '
                'numbers of flows for ' + '; '.join(
                    f'{p} ({_flowcount_diff_summary(flowcount_mismatch[p])})'
                    for p in sorted(flowcount_mismatch)))
        return {'source_file': ref_file, 'key': REFERENCE_KEY,
                'per_protocol': {}, 'protocols_missing': missing,
                'protocols_withheld': list(OOD_PROTOCOLS),
                'reference_backend_mismatch': backend_mismatch,
                'reference_backend_used': ref_backend,
                'reference_backend_reason': ref_backend_reason,
                'run_feature_backend_used': run_backend,
                'reference_flowcount_mismatch': flowcount_mismatch or None,
                'reason': '; '.join(reason_parts)}

    print(f'[REFERENCE-OK] {ref_file}: feature backend {ref_backend!r} matches this run'
          + (' and every per-protocol flow count agrees'
             if run_flow_counts is not None else '')
          + '; our-system rows are kept.')
    return {'source_file': ref_file, 'key': REFERENCE_KEY,
            'per_protocol': per_proto, 'protocols_missing': missing,
            'reference_backend_mismatch': False,
            'reference_backend_used': ref_backend,
            'run_feature_backend_used': run_backend,
            'reference_flowcount_mismatch': None}


def _print_table(title, rows, n_attack_by_proto, aggregated=False):
    keys = ('auc', 'recall', 'precision', 'f1', 'fpr')
    print(f'=== {title} ===')
    for proto in OOD_PROTOCOLS:
        print(f'--- {proto}  (n_benign_test={N_TEST}, n_attack={n_attack_by_proto[proto]})')
        if aggregated:
            print(f'{"method":<13}' + ''.join(k.rjust(18) for k in keys))
            for method in METHODS:
                r = rows[proto][method]
                cells = ''.join(f'{r[k + "_mean"]:.4f}±{r[k + "_std"]:.3f}'.rjust(18)
                                for k in keys)
                print(f'{method:<13}{cells}')
        else:
            print(f'{"method":<13}{"auc":>8}{"recall":>9}{"prec":>8}{"f1":>8}{"fpr":>8}')
            for method in METHODS:
                r = rows[proto][method]
                print(f'{method:<13}{r["auc"]:>8.4f}{r["recall"]:>9.4f}'
                      f'{r["precision"]:>8.4f}{r["f1"]:>8.4f}{r["fpr"]:>8.4f}')


def select_odin_temperature(logits_benign_val, logits_indomain_attack, y_val):
    """Pick the ODIN temperature on IN-DOMAIN validation data only.

    Validation = the reserved benign calibration split (never used for Mahalanobis
    fitting and disjoint from the 150 benign test flows) as negatives plus the
    in-domain DDoS flows (the in-domain attack class) as positives. The selection
    metric is the in-domain AUC of the ODIN score, i.e. how well the score separates
    in-domain attack from in-domain benign. The five OOD test protocols never enter
    this choice. Ties resolve to the SMALLEST T (strict-improvement rule).
    """
    sweep = {}
    if len(logits_benign_val) == 0 or len(logits_indomain_attack) == 0:
        raise ValueError(
            'ODIN temperature selection needs in-domain validation flows: got '
            f'{len(logits_benign_val)} benign validation and '
            f'{len(logits_indomain_attack)} in-domain attack flows')
    best_t, best_auc = T_SWEEP[0], -np.inf
    for temp in T_SWEEP:
        s = np.concatenate([odin_scores(logits_benign_val, temp),
                            odin_scores(logits_indomain_attack, temp)])
        auc = float(roc_auc_score(y_val, s))
        sweep[odin_key(temp)] = {'in_domain_auc': round(auc, 4),
                                 'n_benign_val': int(len(logits_benign_val)),
                                 'n_indomain_attack': int(len(logits_indomain_attack))}
        if auc > best_auc + 1e-12:
            best_t, best_auc = temp, auc
    for temp in T_SWEEP:
        sweep[odin_key(temp)]['selected'] = (temp == best_t)
    return best_t, round(best_auc, 4), sweep


def _print_reference(reference):
    """Print our system's committed numbers (read verbatim, never recomputed).

    The header names the file that was ACTUALLY read (reference['source_file'], set by
    load_reference from --reference) rather than the module constant REFERENCE_FILE,
    which lied whenever --reference pointed elsewhere.
    """
    source_file = reference.get('source_file') or REFERENCE_FILE
    print(f'=== our system (read from {source_file} key {REFERENCE_KEY}) ===')
    if 'reference_backend_used' in reference:
        print(f'    reference feature backend: '
              f'{reference.get("reference_backend_used")!r} | this run: '
              f'{reference.get("run_feature_backend_used")!r}')
    if not reference.get('per_protocol'):
        print('[WITHHELD] our-system rows are NOT printed and NOT comparable: '
              f'{reference.get("reason") or f"no {REFERENCE_KEY} rows in the reference"}')
        print('[WITHHELD] the baseline tables above are unaffected and still valid; they '
              'must NOT be read as a comparison against our system.')
        return
    print(f'{"protocol":<24}{"method":<17}{"auc":>8}{"recall":>9}{"prec":>8}{"f1":>8}{"fpr":>8}')
    for proto in OOD_PROTOCOLS:
        r = reference['per_protocol'].get(proto, {})
        if r:
            print(f'{proto:<24}{"Fusion-complete":<17}{r["auc"]:>8.4f}{r["recall"]:>9.4f}'
                  f'{r["precision"]:>8.4f}{r["f1"]:>8.4f}{r["fpr"]:>8.4f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--max-flows', type=int, default=50,
                    help='max flows extracted per chunk pcap')
    ap.add_argument('--reference', default=REFERENCE_FILE,
                    help='our-system reference JSON; MUST come from the same feature '
                         'backend as this run (default: %(default)s)')
    ap.add_argument('--independent-calibration', action='store_true',
                    help='select each method\'s threshold on the CALIBRATION benign '
                         'split (N_CALIB=%d flows) and measure on the held-out TEST '
                         'split (N_TEST=%d), instead of selecting it on the test flows '
                         'whose realised FPR is then reported. Off by default so the '
                         'committed comparison is reproduced byte-for-byte; on, this '
                         'is Task 6 of the ICASSP plan (spec 4.3).'
                         % (N_CALIB, N_TEST))
    args = ap.parse_args()

    from nemesys_gnn4id.gnn4id import analyzer as analyzer_module
    from nemesys_gnn4id.gnn4id.analyzer import (FEATURE_BACKEND, GNN4IDAnalyzer,
                                                NFSTREAM_AVAILABLE, last_feature_backend)

    t0 = time.time()
    analyzer = GNN4IDAnalyzer(model_path=MODEL_PATH)
    print(f'[INIT] model={MODEL_PATH} device={analyzer.device}')

    by_source, labels_by_source = chunks_by_source()
    # the benign-fit (Benign(0)) and in-domain-attack (DDos(6)) blocks are built from
    # hardcoded class indices; verify them against the chunk index rather than trusting
    # the index to keep its meaning if it is ever re-cut (raises on any other label)
    assert_source_class_pure(BENIGN_SOURCE, labels_by_source.get(BENIGN_SOURCE, []),
                             BENIGN_CLASS)
    assert_source_class_pure(IN_DOMAIN_ATTACK, labels_by_source.get(IN_DOMAIN_ATTACK, []),
                             IN_DOMAIN_CLASS)
    extracted = {}
    extraction_logs = {}
    for source in SOURCES:
        chunks = by_source.get(source, [])
        print(f'[EXTRACT] {source} ({len(chunks)} chunks, max_flows={args.max_flows})')
        extraction_logs[source] = {'n_chunks': len(chunks), 'backends': [],
                                   'chunks_without_flows': [], 'chunks_without_graphs': []}
        if not chunks:
            print(f'  WARNING: no chunks found for {source}')
            extracted[source] = (np.zeros((0, N_MODEL_CLASSES)), np.zeros((0, 16)))
            continue
        extracted[source] = extract_source(analyzer, chunks, args.max_flows,
                                           extraction_log=extraction_logs[source])
        print(f'  total flows: {len(extracted[source][0])}')
    print(f'[EXTRACT] done in {time.time() - t0:.1f}s')

    # -- feature-backend provenance: read the backend that ACTUALLY ran, after
    # -- extraction. The nfstream and scapy extractors are not interchangeable
    # -- (~49% cell agreement; 12/82 dimensions never agree; flow counts differ),
    # -- so the numbers below are uninterpretable without this record.
    backend, reason = last_feature_backend()
    backend_counts = {}
    for log in extraction_logs.values():
        for entry in log['backends']:
            backend_counts[entry['backend']] = backend_counts.get(entry['backend'], 0) + 1
    backend_meta = {
        'feature_backend_requested': FEATURE_BACKEND,
        'feature_backend_used': backend,
        'feature_backend_reason': reason,
        # NFSTREAM_AVAILABLE is `_NFSTREAM_IMPORTABLE and FEATURE_BACKEND != 'scapy'`,
        # i.e. it conflates two different facts. Recorded separately: in a forced-scapy
        # run (the robustness control this whole machinery exists for) nfstream imports
        # fine, and reporting nfstream_importable=False there would be a lie.
        'nfstream_importable': bool(analyzer_module._NFSTREAM_IMPORTABLE),
        'nfstream_available': bool(NFSTREAM_AVAILABLE),
        'backend_note': (
            'feature_backend_used records which extractor actually produced the 82-dim '
            'features. The nfstream and scapy paths are NOT equivalent (~49% cell '
            'agreement; 12/82 dimensions never agree; even the extracted flow counts '
            'differ), so numbers from different backends must never be compared or mixed.'),
        # feature_backend_used is the LAST call; a mid-run switch would otherwise be
        # invisible, so the per-chunk record is kept too
        'feature_backend_per_chunk_counts': backend_counts,
        'feature_backend_consistent': len(backend_counts) <= 1,
    }
    if len(backend_counts) > 1:
        print('!' * 72)
        print(f'[WARNING] MORE THAN ONE FEATURE BACKEND RAN IN THIS JOB: {backend_counts}')
        print('[WARNING] The two backends are not equivalent; the 82-dim features '
              'are not mixable.')
        print('[WARNING] Recorded in _meta.feature_backend_per_chunk_counts, but this '
              'run should be treated as unusable.')
        print('!' * 72)

    # a source with no flows silently poisons every protocol metric (roc_auc_score on
    # a single class); fail here with a clear message instead. The message carries the
    # per-source extraction summary, because a run that dies here writes no JSON and
    # this traceback is then the only record of what the extractor did.
    empty_sources = [s for s in SOURCES if len(extracted[s][0]) == 0]
    if empty_sources:
        summary = {s: {'n_chunks': extraction_logs[s]['n_chunks'],
                       'n_chunks_without_flows':
                           len(extraction_logs[s]['chunks_without_flows']),
                       'chunks_without_flows':
                           extraction_logs[s]['chunks_without_flows'],
                       'backends': sorted({e['backend'] for e in
                                           extraction_logs[s]['backends']}),
                       'backend_reasons': sorted({e['reason'] for e in
                                                  extraction_logs[s]['backends']}),
                       'n_flows': int(len(extracted[s][0]))}
                   for s in empty_sources}
        raise RuntimeError(
            f'no flows extracted for {empty_sources}; refusing to continue. '
            f'Per-source extraction summary: {summary}')
    if len(extracted[BENIGN_SOURCE][0]) < N_FIT + N_CALIB + N_TEST:
        raise RuntimeError(
            f'only {len(extracted[BENIGN_SOURCE][0])} benign flows extracted but the '
            f'benign split needs N_FIT+N_CALIB+N_TEST={N_FIT + N_CALIB + N_TEST}; the '
            'held-out test split would be short and no longer comparable with the '
            'committed reference partition. A too-small --max-flows is the likely cause.')

    # per-source scores that do not depend on the benign split
    base_scores = {}
    argmax_hist = {}
    for source in SOURCES:
        logits, _embed = extracted[source]
        p = scaled_softmax(logits, 1.0) if len(logits) else np.zeros((0, N_MODEL_CLASSES))
        base_scores[source] = {
            'GNN-raw': 1.0 - p[:, BENIGN_CLASS] if len(p) else np.zeros(0),
            'MSP': msp_scores(p) if len(p) else np.zeros(0),
        }
        for temp in T_SWEEP:
            base_scores[source][odin_key(temp)] = (odin_scores(logits, temp)
                                                   if len(logits) else np.zeros(0))
        if len(logits):
            counts = np.bincount(logits.argmax(axis=1), minlength=N_MODEL_CLASSES)
            argmax_hist[source] = {ATTACK_TYPES[c]['name_en']: int(counts[c])
                                   for c in range(N_MODEL_CLASSES) if counts[c]}

    n_benign = len(extracted[BENIGN_SOURCE][0])
    per_seed = {}
    mahalanobis_fit_info = {}
    odin_selection = {}
    n_attack_by_proto = {}
    variant_wins = {proto: {v: 0 for v in MAHALANOBIS_VARIANTS} for proto in OOD_PROTOCOLS}
    variant_wins_f1 = {proto: {v: 0 for v in MAHALANOBIS_VARIANTS} for proto in OOD_PROTOCOLS}
    for seed in args.seeds:
        fit_idx, calib_idx, test_idx = split_benign(n_benign, seed)

        # Mahalanobis: BOTH variants, same in-domain fit flows, different labels
        fits = fit_mahalanobis_variants(
            extracted[BENIGN_SOURCE][1][fit_idx], extracted[IN_DOMAIN_ATTACK][1],
            extracted[BENIGN_SOURCE][0][fit_idx], extracted[IN_DOMAIN_ATTACK][0])
        mahalanobis_fit_info[str(seed)] = {v: fits[v][2] for v in MAHALANOBIS_VARIANTS}

        # ODIN temperature: chosen on in-domain validation data only
        logits_ben_val = extracted[BENIGN_SOURCE][0][calib_idx]
        logits_indomain = extracted[IN_DOMAIN_ATTACK][0]
        y_val = np.array([0] * len(logits_ben_val) + [1] * len(logits_indomain))
        sel_t, sel_auc, sweep = select_odin_temperature(
            logits_ben_val, logits_indomain, y_val)
        odin_selection[str(seed)] = {'selected_temperature': sel_t,
                                     'selection_metric': 'in-domain AUC (ODIN score)',
                                     'selection_in_domain_auc': sel_auc,
                                     'default_temperature': ODIN_DEFAULT_T,
                                     'default_in_domain_auc':
                                         sweep[odin_key(ODIN_DEFAULT_T)]['in_domain_auc'],
                                     'sweep': sweep}

        scores = {}
        for source in SOURCES:
            scores[source] = dict(base_scores[source])
            for variant in MAHALANOBIS_VARIANTS:
                mu, prec, _info = fits[variant]
                scores[source][MAHALANOBIS_VARIANT_METHOD[variant]] = (
                    mahalanobis_scores(extracted[source][1], mu, prec)
                    if len(extracted[source][1]) else np.zeros(0))
            # headline ODIN = sweep-selected T; ODIN_T1000 = CNN default (transparency)
            scores[source]['ODIN'] = base_scores[source][odin_key(sel_t)]
            scores[source]['ODIN_T1000'] = base_scores[source][odin_key(ODIN_DEFAULT_T)]

        protocols = {}
        for proto in OOD_PROTOCOLS:
            n_attack = len(extracted[proto][0])
            n_attack_by_proto[proto] = n_attack
            y = np.array([0] * len(test_idx) + [1] * n_attack)
            methods = {}
            for method in METHODS:
                if method == MAHALANOBIS_HEADLINE:
                    continue  # filled in below from the two variants
                s = np.concatenate([scores[BENIGN_SOURCE][method][test_idx],
                                    scores[proto][method]])
                # Independent-calibration variant: pick the operating point on the
                # CALIBRATION benign split, then measure on the test flows. Without the
                # flag thr_scores stays None and the historical path is untouched.
                thr_src = (scores[BENIGN_SOURCE][method][calib_idx]
                           if args.independent_calibration else None)
                methods[method] = metrics_matched_fpr(s, y, thr_scores=thr_src)
            # headline Mahalanobis = the STRONGER variant on THIS protocol/seed.
            # Selecting per protocol hands the baseline its best variant, which is the
            # conservative (pro-our-system-margin) choice; the rule is stated in
            # _meta['mahalanobis']['headline_rule'] and both variants stay reported.
            winner = max(MAHALANOBIS_VARIANTS,
                         key=lambda v: (methods[MAHALANOBIS_VARIANT_METHOD[v]]['auc'],
                                        methods[MAHALANOBIS_VARIANT_METHOD[v]]['f1']))
            winner_f1 = max(MAHALANOBIS_VARIANTS,
                            key=lambda v: methods[MAHALANOBIS_VARIANT_METHOD[v]]['f1'])
            methods[MAHALANOBIS_HEADLINE] = dict(
                methods[MAHALANOBIS_VARIANT_METHOD[winner]],
                variant=winner, selected_by='test AUC (ties -> higher F1)')
            variant_wins[proto][winner] += 1
            variant_wins_f1[proto][winner_f1] += 1
            protocols[proto] = {'n_benign_test': int(len(test_idx)),
                                'n_attack': int(n_attack), 'methods': methods,
                                'mahalanobis_variant': winner,
                                'mahalanobis_variant_by_f1': winner_f1}
        per_seed[str(seed)] = {'protocols': protocols,
                               'n_fit': int(len(fit_idx)),
                               'n_benign_test': int(len(test_idx)),
                               'mahalanobis_fit': mahalanobis_fit_info[str(seed)],
                               'odin_temperature_selection': odin_selection[str(seed)]}

        seed_rows = {p: {m: protocols[p]['methods'][m] for m in METHODS}
                     for p in OOD_PROTOCOLS}
        _print_table(f'seed {seed}', seed_rows, n_attack_by_proto)
        for variant in MAHALANOBIS_VARIANTS:
            print(f'  mahalanobis[{variant}] fit: {mahalanobis_fit_info[str(seed)][variant]}')
        print('  mahalanobis headline variant per protocol: '
              + ', '.join(f'{p}={protocols[p]["mahalanobis_variant"]}'
                          for p in OOD_PROTOCOLS))
        print(f'  ODIN temperature: selected T={sel_t:g} '
              f'(in-domain AUC {sel_auc:.4f}); default T={ODIN_DEFAULT_T:g} '
              f'(in-domain AUC '
              f'{odin_selection[str(seed)]["default_in_domain_auc"]:.4f})')
        print(f'  ODIN sweep (in-domain AUC by T): '
              + ', '.join(f'{odin_key(t)}={sweep[odin_key(t)]["in_domain_auc"]:.4f}'
                          for t in T_SWEEP))

    mean_std = aggregate(per_seed)
    _print_table(f'mean/std over seeds {args.seeds}', mean_std, n_attack_by_proto,
                 aggregated=True)

    # our system's rows are READ from a reference file and juxtaposed with the baselines
    # COMPUTED above, so the reference is verified against this run's actual backend and
    # per-protocol flow counts. Both are threaded in from here: `backend` was read from
    # the analyzer after extraction (above), and the flow counts come from this run's own
    # per-protocol records (seed-independent by construction: n_attack = extracted attack
    # flows, n_benign_test = the held-out benign test split size).
    run_flow_counts = {
        proto: {'n_attack': int(n_attack_by_proto[proto]),
                'n_benign_test': int(
                    per_seed[str(args.seeds[0])]['protocols'][proto]['n_benign_test'])}
        for proto in OOD_PROTOCOLS
    }
    reference = load_reference(args.reference, run_backend=backend,
                               run_flow_counts=run_flow_counts)
    if reference:
        _print_reference(reference)
    # A withheld our-system block is a FIRST-CLASS _meta fact, not something buried in
    # our_system_provenance: the baseline numbers in this JSON are still valid, but nobody
    # may read them as a comparison against our system.
    reference_rows_withheld = bool(reference) and not reference.get('per_protocol')
    if reference_rows_withheld:
        print('=' * 78)
        print('[WARNING] THIS RUN HAS NO OUR-SYSTEM ROWS: the tables above compare '
              'baselines only.')
        print('[WARNING] reason: '
              + (reference.get('reason') or f'no {REFERENCE_KEY} rows in the reference'))
        print('=' * 78)

    out = {
        '_meta': {
            'purpose': ('main-line comparison of the protocol-aware system against '
                        'standard OOD detection methods on a shared GNN4ID base model, '
                        'per protocol, matched ~20% FPR'),
            'model': MODEL_PATH,
            'device': str(analyzer.device),
            'base_embedding': 'graph_prediction_1 output (16-dim penultimate)',
            'classes': {'benign_index': BENIGN_CLASS, 'in_domain_attack_index': IN_DOMAIN_CLASS,
                        'in_domain_attack': IN_DOMAIN_ATTACK, 'n_model_classes': N_MODEL_CLASSES},
            'methods': METHODS,
            'target_fpr': TARGET_FPR,
            'threshold_rule': ('multisignal_label_free_fusion.threshold_at_target_fpr '
                               'semantics, copied here: benign-only candidates just above '
                               'each distinct benign score, operating point = closest '
                               'realised FPR to 0.20, detection strict ">". '
                               'pu_two_step_detection.metrics_at_fpr was NOT used because '
                               'its internal np.quantile rule is not tie-aware.'),
            'benign_scores_for_threshold': (
                'each method\'s own scores on the held-out benign CALIBRATION flows '
                '(N_CALIB=%d), while the reported FPR is measured on the held-out TEST '
                'flows (N_TEST=%d). Threshold selection and measurement are therefore '
                'disjoint, so the realised FPR is whatever the method actually achieves '
                'rather than a value pinned to the target by construction.'
                % (N_CALIB, N_TEST)
                if args.independent_calibration else
                'each method\'s own scores on the held-out benign TEST flows (y == 0), '
                'the convention of multisignal_label_free_fusion.py and '
                'complete_fusion_controlled_fpr.py. NOTE: with this convention the '
                'operating point is chosen on the same flows whose realised FPR is '
                'reported, so that FPR sits near the 0.20 target by construction. Use '
                '--independent-calibration to separate the two.'),
            'independent_calibration': bool(args.independent_calibration),
            'split_convention': {'source': 'experiments/multisignal_label_free_fusion.py',
                                 'split_benign': 'copied (N_FIT, N_CALIB, N_TEST)',
                                 'N_FIT': N_FIT, 'N_CALIB': N_CALIB, 'N_TEST': N_TEST,
                                 'seeds': args.seeds},
            'mahalanobis': {
                'fit_data': ('in-domain training data only: the benign fit split '
                             f'({N_FIT} flows) plus every extracted in-domain DDoS flow '
                             f'({IN_DOMAIN_ATTACK}); no OOD test protocol enters the fit'),
                'variants': {
                    'known': ('textbook Lee et al. fit on the KNOWN in-domain labels '
                              '(benign-fit -> Benign(0), in-domain DDoS -> DDos(6)); the '
                              'in-domain domain supplies only these two classes, so K=2 '
                              'class-conditional Gaussians are fitted'),
                    'argmax': ('SAME fit flows, labels = the base model\'s own argmax '
                               'prediction (label-free: needs no in-domain attack labels), '
                               'so K = the number of distinct predicted classes'),
                },
                'headline_rule': ('the reported per-protocol "Mahalanobis" is the STRONGER '
                                  'of the two variants for that protocol AND seed, chosen '
                                  'by test AUC (ties -> higher F1); this oracle choice over '
                                  'the variant favours the baseline and therefore keeps our '
                                  'margin conservative. Both variants are reported '
                                  'separately (Mahalanobis_known / Mahalanobis_argmax) and '
                                  'the per-protocol winner is recorded as '
                                  'mahalanobis_variant, so the choice is auditable.'),
                'fit_contract': ('fit_mahalanobis is called ONCE per seed per variant on the '
                                 'whole in-domain split (never per flow, which would raise '
                                 '"class 0 has no training samples"), with contiguous '
                                 'integer labels in [0, K) after the documented remap'),
                'label_index_convention': ('variant labels are MODEL class indices in '
                                           '[0, 8) (Benign=0, ..., BruteForce=7) and a label '
                                           'outside that range raises. Because in-domain '
                                           'data covers only a subset of the 8 classes, the '
                                           'labels present are remapped to a contiguous '
                                           '0..K-1 fit index before fit_mahalanobis (which '
                                           'requires contiguity, commit 0c3bb6d); the remap '
                                           'is recorded per variant per seed in '
                                           'label_remap_to_fit_index, so no class is '
                                           'silently dropped from the pooled covariance'),
                'score_convention': ('SQUARED Mahalanobis distance to the nearest '
                                     'class-conditional Gaussian (no square root; unlike '
                                     'scipy.spatial.distance.mahalanobis). Only the '
                                     'ranking/threshold matter, and the convention is used '
                                     'consistently for every method'),
                'embedding_space': ('16-dim graph_prediction_1 penultimate embedding, used '
                                    'as-is: it needs no standardisation (the 82-dim raw '
                                    'flow-feature space would, since cond(cov) is ~1.3e18 '
                                    'unstandardised vs ~19 after z-scoring, which makes the '
                                    '1e-6 ridge inert); no feature-space fallback was '
                                    'used'),
                'variant_wins_by_auc_per_protocol': variant_wins,
                'variant_wins_by_f1_per_protocol': variant_wins_f1,
                'per_seed_detail': mahalanobis_fit_info},
            'odin_temperature': {
                'sweep': T_SWEEP,
                'default_cnn_temperature': ODIN_DEFAULT_T,
                'default_reported_as': 'ODIN_T1000',
                'headline_method': ('ODIN uses the sweep-selected temperature; selection '
                                    'data = in-domain validation only (benign calibration '
                                    'split as negatives + in-domain DDoS as positives), '
                                    'metric = in-domain AUC of the ODIN score, ties -> '
                                    'smallest T. The five OOD test protocols never enter '
                                    'the selection.'),
                'why': ('T=1000 is Liang et al.\'s CNN-scale default and is nearly '
                        'degenerate for a C=8 classifier (MSP ceiling 1 - 1/8 = 0.875, '
                        'in-domain score spread ~1.1e-4 at T=1000 vs ~1.5e-1 at T=1), so '
                        'reporting it alone would cripple the baseline and make the '
                        'comparison unfair in our favour'),
                'selected_temperature_per_seed': {s: v['selected_temperature']
                                                  for s, v in odin_selection.items()},
                'selection_detail_per_seed': odin_selection},
            'max_flows_per_chunk': args.max_flows,
            'chunk_dir': os.path.relpath(CHUNK_DIR, REPO).replace(os.sep, '/'),
            'n_flows_per_source': {s: int(len(extracted[s][0])) for s in SOURCES},
            'extraction': {s: {'n_chunks': extraction_logs[s]['n_chunks'],
                               'n_chunks_without_flows':
                                   len(extraction_logs[s]['chunks_without_flows']),
                               'chunks_without_flows':
                                   extraction_logs[s]['chunks_without_flows'],
                               'chunks_without_graphs':
                                   extraction_logs[s]['chunks_without_graphs'],
                               'n_flows': int(len(extracted[s][0])),
                               'backends': sorted({e['backend'] for e in
                                                   extraction_logs[s]['backends']})}
                          for s in SOURCES},
            'model_argmax_histogram_per_source': argmax_hist,
            'our_system_provenance': reference,
            # reference/run provenance gate -- see load_reference(). When the reference's
            # recorded feature backend (or its flow counts) disagrees with this run's, the
            # our-system rows are withheld and these fields say so at the top level of
            # _meta: the baseline numbers stay valid, but nothing here may be read as a
            # comparison against our system.
            'reference_backend_mismatch': bool(
                reference and reference.get('reference_backend_mismatch')),
            'reference_flowcount_mismatch': (
                reference.get('reference_flowcount_mismatch') if reference else None),
            'reference_rows_withheld': reference_rows_withheld,
            'reference_rows_withheld_reason': (
                (reference.get('reason') or f'no {REFERENCE_KEY} rows in the reference')
                if reference_rows_withheld else None),
            'our_system_reference_caveat': ('Fusion-complete is a single committed partition '
                                            '(benign train 350 = N_FIT+N_CALIB, test 150, '
                                            'seed 42) and its threshold rule is the plain '
                                            'quantile in complete_fusion_controlled_fpr.py; '
                                            'our seed-42 benign test split is the identical '
                                            '150 flows, so the comparison is matched.'),
            'notes': ('feature extraction and GNN inference run once per source and are '
                      'reused across seeds; GNN-raw score = 1 - P(benign) of the model\'s '
                      'own softmax; realised fpr is reported, not assumed'),
            'elapsed_seconds': None,
        },
        'per_seed': per_seed,
        'mean_std': mean_std,
        'reference_our_system': reference,
    }

    # recorded AFTER extraction (read from the analyzer, never assumed). See
    # backend_meta above; feature_backend_used is the extractor that actually produced
    # the 82-dim features behind every number in this file.
    out['_meta'].update(backend_meta)
    # the per-chunk backend record is bulky but is the only proof of what ran
    out['_meta']['feature_backend_per_chunk'] = {
        s: extraction_logs[s]['backends'] for s in SOURCES}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out['_meta']['elapsed_seconds'] = round(time.time() - t0, 1)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'Saved: {args.out}  (total {out["_meta"]["elapsed_seconds"]}s)')


if __name__ == '__main__':
    main()
