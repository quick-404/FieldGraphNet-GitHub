import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from mdiplier_utils import parse_mdiplier_csv, splits_to_segments, merge_header_body, collect_mdiplier_messages

SAMPLE = (
    '\ufeffHexstream,Split Indexes,Splited Hexstream\n'
    'ab26010000010000000000000474696d65046e69737403676f760000010001,'
    '"[0, 2, 4, 6, 8, 24, 26]",ab 26 01 00\n'
    'ab27010000010000000000000474696d65046e69737403676f760000010001,'
    '"[0, 2, 4, 26, 28]",ab 27 01 00\n'
)


def _write_sample(path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(SAMPLE)


def test_parse_mdiplier_csv():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'h.out')
        _write_sample(p)
        recs = parse_mdiplier_csv(p)
    assert len(recs) == 2
    assert recs[0]['hex'].startswith('ab26')
    assert recs[0]['splits'] == [0, 2, 4, 6, 8, 24, 26]


def test_splits_to_segments():
    hexstream = 'ab260100'
    segments = splits_to_segments(hexstream, [0, 2, 4])
    assert len(segments) == 2
    assert segments[0] == {'offset': 0, 'length': 2, 'bytes': bytes([0xab, 0x26])}
    assert segments[1] == {'offset': 2, 'length': 2, 'bytes': bytes([0x01, 0x00])}


def test_merge_header_body_union():
    header = [{'hex': 'aabb', 'splits': [0, 2, 4]}]
    body = [{'hex': 'aabb', 'splits': [0, 1, 4]}]
    merged = merge_header_body(header, body)
    assert len(merged) == 1
    assert merged[0]['splits'] == [0, 1, 2, 4]


def test_run_mdiplier_produces_merged_messages():
    pcap = r'D:/网络流量分析项目/nemesys-gnn4id/data/data/dns_100.pcap'
    if not os.path.exists(pcap):
        return  # skip if data missing
    with tempfile.TemporaryDirectory() as d:
        msgs = collect_mdiplier_messages(pcap, d)
    assert len(msgs) > 0
    assert 'hex' in msgs[0]
    assert 'segments' in msgs[0]
    assert msgs[0]['segments'][0]['length'] > 0


from mdiplier_utils import align_by_hex


def test_align_by_hex():
    bcdg = [{'hex': 'aabb', 'segments': []}, {'hex': 'ccdd', 'segments': []}, {'hex': 'eeff', 'segments': []}]
    mdi = [{'hex': 'aabb', 'segments': []}, {'hex': 'ccdd', 'segments': []}]
    b_out, m_out, rate = align_by_hex(bcdg, mdi)
    assert len(b_out) == 2 and len(m_out) == 2
    assert abs(rate - 2 / 3) < 1e-6
