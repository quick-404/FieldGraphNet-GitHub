"""Supervised structural-fingerprint classifier: fair comparison vs SVM.

Runs NEMESYS BCDG on each chunk, extracts the 24-dim structural fingerprint
per message, aggregates per flow (mean), trains a supervised SVM (RBF) on
in-domain flow-level structural vectors (Benign=0 vs DDoS=attack), and tests
per OOD protocol. This uses labels like SVM does, but on OUR structural
fingerprint features instead of 82-dim flow statistics.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from sklearn.svm import SVC
from nemesys_gnn4id.nemesys.analyzer import NEMESYSAnalyzer
from nemesys_gnn4id.nemesys.cluster import extract_structural_fingerprint
from nemesys_gnn4id.pipeline import canonical_flow_tuple

CHUNK_DIR = 'data/data/23pcap_chunks'
IN_DOMAIN = ['BenignTraffic.pcap', 'DDoS-HTTP_Flood-.pcap']
OOD_PROTOCOLS = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
                 'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']


def flow_structural_vectors():
    """Return {source: {flow_key: mean 24-dim fingerprint}}."""
    analyzer = NEMESYSAnalyzer()
    with open(os.path.join(CHUNK_DIR, '_chunk_index.json'), encoding='utf-8') as f:
        chunks = json.load(f)
    out = defaultdict(dict)
    n_chunks = 0
    for chunk_name, (_label, src) in chunks.items():
        chunk_path = os.path.join(CHUNK_DIR, chunk_name)
        if not os.path.exists(chunk_path):
            continue
        result = analyzer.analyze(chunk_path)
        segs = result.get('segments', [])
        # accumulate fingerprints per flow
        per_flow_fp = defaultdict(list)
        for seg in segs:
            if not isinstance(seg, dict) or 'src' not in seg:
                continue
            key = canonical_flow_tuple(seg.get('src', ''), seg.get('dst', ''),
                                       seg.get('sport', 0), seg.get('dport', 0),
                                       seg.get('proto', 0))
            fp = extract_structural_fingerprint(seg)
            per_flow_fp[key].append(fp)
        for key, fps in per_flow_fp.items():
            out[src][key] = np.mean(fps, axis=0)  # mean over the flow's messages
        n_chunks += 1
    print(f'Processed {n_chunks} chunks')
    return out


CACHE = '/tmp/supstruct_flows.json'


def load_flows():
    """Load cached flow vectors if present, else compute and cache."""
    if os.path.exists(CACHE):
        with open(CACHE, encoding='utf-8') as f:
            raw = json.load(f)
        return {src: {tuple(map(int, k.split(','))): np.array(v, dtype=np.float32)
                      for k, v in flows.items()}
                for src, flows in raw.items()}
    flows = flow_structural_vectors()
    save = {src: {','.join(map(str, k)): v.tolist() for k, v in fv.items()}
            for src, fv in flows.items()}
    with open(CACHE, 'w', encoding='utf-8') as f:
        json.dump(save, f)
    return flows


def main():
    flows = load_flows()
    # binary: benign=0, DDoS=1 (in-domain labels)
    n_benign = len(flows.get('BenignTraffic.pcap', {}))
    n_ddos = len(flows.get('DDoS-HTTP_Flood-.pcap', {}))
    X_in = np.array([fp for s in IN_DOMAIN for fp in flows.get(s, {}).values()])
    y_in = np.array([0] * n_benign + [1] * n_ddos)
    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_in, y_in)
    print(f'Trained supervised structural SVM on {len(X_in)} in-domain flows '
          f'(benign={n_benign}, ddos={n_ddos})')

    # per-protocol recall on OOD
    with open('eval_results/fusion_comparison/per_protocol_baselines.json', encoding='utf-8') as f:
        baselines = json.load(f)
    print(f'{"protocol":<24s} {"supStruct":>10s} {"SVM(82dim)":>11s}  supStruct>=SVM')
    for ood in OOD_PROTOCOLS:
        fvecs = [fp for fp in flows.get(ood, {}).values()]
        if not fvecs:
            print(f'{ood}: no flows')
            continue
        Xt = np.array(fvecs)
        pred = (clf.predict(Xt) == 1)  # 1 = attack
        rec = float(pred.sum() / len(pred))
        svm_rec = baselines[ood.replace('.pcap', '')]['SVM']['recall']
        print(f'{ood:<24s} {rec:>10.3f} {svm_rec:>11.3f}  {rec >= svm_rec}')


if __name__ == '__main__':
    main()
