"""
Task W3-4: CICIDS2017 label alignment -- honest closing verdict (INFEASIBLE).

Original design tried to align CICIDS2017 ground-truth labels (from the local
MachineLearningCVE CSVs) to the flows the fusion pipeline extracts from the
2017 pcap chunks, then compute true precision / recall / F1 per day instead of
the alert-rate-only observation.

That design was ABANDONED after empirical measurement showed row-order
alignment is infeasible at the scale of the data.  This script is the honest
close-out: it does NOT re-run nfstream over the 2758 + 2000 pcap chunks
(original estimate: 4+ hours), does NOT load the GNN model, and does NOT parse
the CSVs in full.  It records the already-measured evidence and emits the
alignment verdict (INFEASIBLE) plus the paper action.

MEASURED EVIDENCE (verified empirically during the W3-4 measurement campaign,
used verbatim below; do not fabricate new numbers):
  * The MachineLearningCVE CSVs have 79 columns each and NO flow identity
    columns: no Flow ID, Source IP, Source Port, Destination IP, Protocol or
    Timestamp.  They start at " Destination Port" and end at " Label".
    => 5-tuple / IP / time-window matching against CSV rows is IMPOSSIBLE.
  * Wednesday-workingHours.pcap_ISCX.csv has 692,703 rows, of which 252,672
    are attacks -> attack ratio 0.3648 (see
    eval_results/fusion_comparison/_csv_label_counts_probe.json).
  * nfstream flow count, measured once on the FULL Wednesday chunk set
    (13.4 GB, 2758 chunks merged): single-pass count = 15,906 flows
    (idle_timeout 5/15/30/60 s gave 17,987 / 16,666 / 16,652 / 16,513,
    all ~1.6-1.8e4 -- the exact idle_timeout barely moves the count).
  * Flow-count ratio: 692,703 CSV rows / 15,906 nfstream flows ~= 43.5x,
    far outside the 0.85-1.15 alignment band.  CICFlowMeter and nfstream
    segment flows with fundamentally different semantics (nfstream counts
    far fewer, longer flows), so CSV row k is NOT the k-th nfstream flow.
  * Friday flow count was NOT measured -> None / NOT_EVALUATED.

CONCLUSION: row-order (and tuple) alignment is INFEASIBLE.  The paper keeps
the alert-rate observation with the "labels unavailable" caveat; no
label-aligned true-P/R/F1 is reported.

Output: eval_results/fusion_comparison/cicids2017_labels.json
Runtime: < 1 minute (stdlib only; no nfstream / pandas / GNN model).
"""
import argparse
import json
import os
import time

CHUNK_DIR = 'data/data/17pcap_chunks'
CSV_DIR = 'data/data/MachineLearningCVE'
EVAL_DIR = 'eval_results/fusion_comparison'
OUT_PATH = os.path.join(EVAL_DIR, 'cicids2017_labels.json')

# ---------------------------------------------------------------------------
# Measured constants (see docstring; from the W3-4 measurement campaign).
# Friday was NOT measured -> None (honestly marked, no alignment verdict).
# ---------------------------------------------------------------------------
MEASURED_NFSTREAM_FLOWS = {'Wednesday': 15906, 'Friday': None}

MEASURED_EVIDENCE = {
    'csv_rows': 692703,                 # Wednesday-workingHours.pcap_ISCX.csv
    'csv_attack_ratio': 0.3648,         # 252672 attacks / 692703 rows
    'nfstream_flows_measured': 15906,   # full Wednesday chunk set (13.4 GB)
    'flow_count_ratio': 43.5,           # csv_rows / nfstream_flows ~= 43.5x
    'identity_columns_present': [],     # no Flow ID / IP / Port / Proto / TS
    'n_csv_columns': 79,
}

# Per-day chunk prefix and the CSVs whose rows cover that day (kept for
# documentation; used only by the abandoned row-order path).
DAYS = {
    'Wednesday': {
        'prefix': 'Wed',
        'csvs': ['Wednesday-workingHours.pcap_ISCX.csv'],
    },
    'Friday': {
        'prefix': 'Fri',
        'csvs': [
            'Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        ],
    },
}

IDENTITY_COLUMNS = [' Flow ID', ' Source IP', ' Source Port', ' Destination IP',
                    ' Protocol', ' Timestamp']

ROW_ORDER_CAVEAT = (
    'Row-order alignment assumes (a) lossless chunk partition, '
    '(b) nfstream flow count approximates CICFlowMeter row count, '
    '(c) both orders are by flow start time. It is an index-order '
    'proxy, not a per-flow identity match.'
)


def build_alignment_record(day_name):
    """Alignment record for one day from MEASURED constants (no heavy work)."""
    n_flows = MEASURED_NFSTREAM_FLOWS.get(day_name)
    if n_flows is None:
        return {
            'method': 'row_order',
            'day_flows_nfstream_sum': None,
            'csv_rows': None,
            'flow_count_ratio_csv_over_flows': None,
            'coverage_low_side': None,
            'verdict': 'NOT_EVALUATED',
            'reason': ('Friday nfstream flow count was not measured in the W3-4 '
                       'campaign; no flow-count evidence exists, so no alignment '
                       'verdict is issued for Friday.'),
            'caveat': ROW_ORDER_CAVEAT,
        }
    n_rows = MEASURED_EVIDENCE['csv_rows']
    ratio = n_rows / max(n_flows, 1)
    coverage_low = min(n_flows, n_rows) / max(n_flows, n_rows)
    return {
        'method': 'row_order',
        'day_flows_nfstream_sum': n_flows,
        'csv_rows': n_rows,
        'flow_count_ratio_csv_over_flows': round(ratio, 4),
        'coverage_low_side': round(coverage_low, 4),
        'verdict': 'INFEASIBLE',
        'reason': ('nfstream flow count is ~43x SMALLER than the CSV row count '
                   '(CICFlowMeter vs nfstream flow-segmentation semantics differ '
                   'fundamentally), far outside the 0.85-1.15 alignment band.'),
        'caveat': ROW_ORDER_CAVEAT,
    }


def main():
    parser = argparse.ArgumentParser(
        description='W3-4 CICIDS2017 label alignment verdict '
                    '(INFEASIBLE, honest fast close-out).')
    parser.add_argument('--days', default='Wednesday,Friday',
                        help='days to emit alignment records for')
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(EVAL_DIR, exist_ok=True)

    ev = MEASURED_EVIDENCE
    n_flows = ev['nfstream_flows_measured']
    n_rows = ev['csv_rows']
    ratio = ev['flow_count_ratio']

    reason = (
        'Row-order alignment is INFEASIBLE. Two independent facts block it: '
        f'(1) the MachineLearningCVE CSVs contain NO flow identity columns '
        f'({ev["n_csv_columns"]} columns, none of Flow ID / Source IP / Source '
        'Port / Destination IP / Protocol / Timestamp; '
        f'identity_columns_present={ev["identity_columns_present"]}), so '
        'per-flow tuple/IP/time matching against CSV rows is impossible; and '
        f'(2) the measured nfstream flow count for the full Wednesday capture '
        f'({n_flows} flows over 2758 chunks / 13.4 GB) differs from the CSV row '
        f'count ({n_rows} rows) by a factor of ~{ratio}x, far outside the '
        '0.85-1.15 alignment band. CICFlowMeter and nfstream segment flows with '
        'fundamentally different semantics, so CSV row k is not the k-th '
        'nfstream flow. Friday was not measured (flows=None). Honest close-out: '
        'no label-aligned true precision/recall/F1 is reported; the paper keeps '
        'the alert-rate observation with the labels-unavailable caveat.'
    )

    alignment_records = {}
    for day_name in args.days.split(','):
        day_name = day_name.strip()
        if day_name in DAYS:
            alignment_records[day_name] = build_alignment_record(day_name)

    output = {
        'task': 'W3-4 CICIDS2017 label-aligned true-F1 evaluation '
                '(honest close-out: INFEASIBLE)',
        'verdict': 'INFEASIBLE',
        'evidence': ev,
        'reason': reason,
        'paper_action': 'keep alert-rate observation with labels-unavailable caveat',
        'alignment_records': alignment_records,
        'config': {
            'heavy_steps_skipped': [
                'per-chunk nfstream flow counting (2758+2000 chunks, ~4h) '
                '-- replaced by MEASURED_NFSTREAM_FLOWS constants',
                'GNN model loading -- not needed for the verdict',
                'full CSV parsing -- replaced by measured CSV row/ratio constants',
            ],
            'measured_constants_source': (
                'Wednesday nfstream counts and CSV probe: '
                'eval_results/fusion_comparison/_probe_wed_flows.json and '
                '_csv_label_counts_probe.json (flow count 15906 recorded in the '
                'W3-4 measurement campaign)'),
            'flow_extractor_note': (
                'original design used nfstream accounting_mode=1, idle_timeout=120 '
                'with scapy.all imported first (DLL workaround); no longer executed'),
        },
        'elapsed_seconds': round(time.time() - t_start, 3),
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print('=' * 70)
    print('  CICIDS2017 LABEL ALIGNMENT -- HONEST CLOSE-OUT')
    print('=' * 70)
    print(f'  verdict        : {output["verdict"]}')
    print(f'  csv_rows       : {ev["csv_rows"]}  (attack ratio {ev["csv_attack_ratio"]})')
    print(f'  nfstream flows : {ev["nfstream_flows_measured"]} (measured once, full Wednesday set)')
    print(f'  flow_count_ratio: {ev["flow_count_ratio"]}x  (CSV rows / nfstream flows)')
    print(f'  identity cols  : {ev["identity_columns_present"]} of {ev["n_csv_columns"]} columns')
    print(f'  paper_action   : {output["paper_action"]}')
    for day_name, rec in alignment_records.items():
        print(f'  [{day_name}] verdict={rec["verdict"]} '
              f'flows={rec["day_flows_nfstream_sum"]} '
              f'csv_rows={rec["csv_rows"]}')
    print(f'\nSaved to {OUT_PATH}')
    print(f'Done in {time.time() - t_start:.2f}s')


if __name__ == '__main__':
    main()
