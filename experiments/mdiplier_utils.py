import csv
import json
import os
import subprocess
import sys


PROTOCOLS_6 = ['dhcp', 'dns', 'dnp3', 'modbus', 'ntp', 's7comm']
MDIPLIER_ROOT = r'D:/网络流量分析项目/nemesys-gnn4id/mdi方法对比/MDIplier'


def parse_mdiplier_csv(path):
    """Parse an MDIplier header/body .out CSV into per-message dicts.

    Returns:
        list of {'hex': str, 'splits': list[int]}
    """
    records = []
    with open(path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'hex': row['Hexstream'].strip(),
                'splits': [int(x) for x in json.loads(row['Split Indexes'])],
            })
    return records


def splits_to_segments(hexstream, splits):
    """Convert MDIplier split indexes into field segments.

    Args:
        hexstream: message payload as a hex string
        splits: list of byte offsets (field boundaries)

    Returns:
        list of {'offset': int, 'length': int, 'bytes': bytes}
    """
    data = bytes.fromhex(hexstream)
    boundaries = sorted(set(splits))
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end <= start:
            continue
        segments.append({
            'offset': start,
            'length': end - start,
            'bytes': data[start:end],
        })
    return segments


def merge_header_body(header_records, body_records):
    """Union the header and body split indexes per message, keyed by hex.

    Both CSVs list the same messages (same hexstream); the header and body
    segmentations differ. The union gives the finest field structure.
    """
    by_hex = {}
    for rec in header_records + body_records:
        key = rec['hex'].lower()
        by_hex.setdefault(key, set()).update(rec['splits'])
    return [{'hex': h, 'splits': sorted(s)} for h, s in by_hex.items()]


def run_mdiplier(pcap_path, workdir):
    """Run MDIplier on one pcap, return (header_records, body_records).

    Uses sys.executable (the current interpreter) so the broken .venv314
    re-exec in mdiplier/main.py is bypassed: the venv python.exe has been
    renamed and `ensure_project_python()` falls back to the current python.
    """
    os.makedirs(workdir, exist_ok=True)
    header_path = os.path.join(workdir, 'header.out')
    body_path = os.path.join(workdir, 'body.out')
    tmp_dir = os.path.join(workdir, 'tmp')
    cmd = [
        sys.executable, 'mdiplier/main.py',
        '-i', os.path.abspath(pcap_path),
        '-o', tmp_dir,
        '-hr', header_path,
        '-br', body_path,
    ]
    result = subprocess.run(cmd, cwd=MDIPLIER_ROOT, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f'MDIplier failed: {result.stderr.decode(errors="replace")[-500:]}')
    return parse_mdiplier_csv(header_path), parse_mdiplier_csv(body_path)


def collect_mdiplier_messages(pcap_path, workdir):
    """Run MDIplier on a pcap and return per-message field segments.

    Returns:
        list of {'hex': str, 'segments': list[{offset, length, bytes}]}
    """
    header_records, body_records = run_mdiplier(pcap_path, workdir)
    merged = merge_header_body(header_records, body_records)
    messages = []
    for rec in merged:
        messages.append({
            'hex': rec['hex'],
            'segments': splits_to_segments(rec['hex'], rec['splits']),
        })
    return messages


def align_by_hex(bcdg_messages, mdiplier_messages):
    """Match messages by normalized payload hex.

    Returns (aligned_bcdg, aligned_mdiplier, match_rate) where match_rate is
    the fraction of BCDG messages that were matched.
    """
    mdi_by_hex = {}
    for m in mdiplier_messages:
        mdi_by_hex.setdefault(m['hex'].lower(), m)

    aligned_bcdg = []
    aligned_mdi = []
    matched = 0
    for m in bcdg_messages:
        target = mdi_by_hex.get(m.get('payload_hex', m.get('hex', '')).lower())
        if target is None:
            continue
        aligned_bcdg.append(m)
        aligned_mdi.append(target)
        matched += 1

    total = max(len(bcdg_messages), 1)
    return aligned_bcdg, aligned_mdi, matched / total
