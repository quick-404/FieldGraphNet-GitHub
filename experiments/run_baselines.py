"""Per-protocol OOD comparison: train on in-domain, test on each OOD protocol."""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
CHUNK_DIR = 'data/data/23pcap_chunks'
MODEL_PATH = 'models/model.pth'
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def compute_metrics(y_true, y_pred):
    n = len(y_true)
    tp = sum(1 for i in range(n) if y_true[i]>0 and y_pred[i]>0)
    fp = sum(1 for i in range(n) if y_true[i]==0 and y_pred[i]>0)
    fn = sum(1 for i in range(n) if y_true[i]>0 and y_pred[i]==0)
    tn = sum(1 for i in range(n) if y_true[i]==0 and y_pred[i]==0)
    return {'recall':round(tp/max(tp+fn,1),4),'fpr':round(fp/max(fp+tn,1),4),'f1':round(2*tp/max(2*tp+fp+fn,1),4)}

print('Extracting features...')
with open(os.path.join(CHUNK_DIR, '_chunk_index.json')) as f:
    chunks = json.load(f)
from nemesys_gnn4id.gnn4id.analyzer import GNN4IDAnalyzer
analyzer = GNN4IDAnalyzer(model_path=MODEL_PATH)

all_features = []; all_labels = []; all_protocols = []
for chunk_name, (attack_label, source_pcap) in chunks.items():
    chunk_path = os.path.join(CHUNK_DIR, chunk_name)
    if not os.path.exists(chunk_path): continue
    df = analyzer.extract_features(chunk_path, max_flows=50)
    if df.empty or 'flow_features' not in df.columns: continue
    for _, row in df.iterrows():
        feat = list(row['flow_features'])
        for z in range(77, min(82, len(feat))): feat[z] = 0.0  # drop ports
        all_features.append(feat)
        all_labels.append(attack_label)
        all_protocols.append(source_pcap)

X = np.array(all_features); y = np.array(all_labels)
print(f'Total: {len(X)} samples')

# Define training (in-domain) and test (OOD) splits
# In-domain: BenignTraffic, DDoS-HTTP_Flood
# OOD per protocol: DNS_Spoofing, Mirai, SqlInjection, BruteForce, Recon
in_domain_pcaps = ['BenignTraffic.pcap', 'DDoS-HTTP_Flood-.pcap']
ood_pcaps = ['DNS_Spoofing.pcap', 'Mirai-udpplain.pcap', 'SqlInjection.pcap',
             'DictionaryBruteForce.pcap', 'Recon-PortScan.pcap']

# Training data = in-domain only
train_idx = [i for i, p in enumerate(all_protocols) if p in in_domain_pcaps]
X_train = X[train_idx]; y_train = y[train_idx]
print(f'Train (in-domain): {len(X_train)} samples')

# Train all classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=4),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB(),
    'C4.5 (Tree)': DecisionTreeClassifier(criterion='entropy', random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)

# Also get GNN4ID baseline from existing eval
import json as j
with open('eval_results/fusion_comparison/nemesys_first_eval.json') as f:
    eval_data = j.load(f)
gnn_by_source = eval_data['by_source']

# Per-protocol evaluation
print('\n' + '='*70)
print('PER-PROTOCOL OOD COMPARISON')
print('(Train on in-domain only → Test on each OOD protocol)')
print('='*70)

for ood_pcap in ood_pcaps:
    print(f'\n--- OOD Protocol: {ood_pcap.replace(".pcap","")} ---')
    ood_idx = [i for i, p in enumerate(all_protocols) if p == ood_pcap]
    if not ood_idx: continue
    X_test = X[ood_idx]; y_test = y[ood_idx]

    print(f'  {"Method":25s} {"Recall":>8s} {"FPR":>8s} {"F1":>8s}')
    print(f'  {"-"*50}')
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        print(f'  {name:25s} {m["recall"]:>8.4f} {m["fpr"]:>8.4f} {m["f1"]:>8.4f}')

    # GNN4ID raw + Fusion from existing eval
    if ood_pcap in gnn_by_source:
        gnn_r = gnn_by_source[ood_pcap]['gnn_raw']
        fus_r = gnn_by_source[ood_pcap]['fusion_alert']
        print(f'  {"GNN4ID (raw)":25s} {gnn_r["recall"]:>8.4f}  —  {gnn_r["f1"]:>8.4f}')
        print(f'  {"FieldGraph-Net (Fusion)":25s} {fus_r["recall"]:>8.4f} {fus_r["fpr"]:>8.4f} {fus_r["f1"]:>8.4f}')

# Save results to file
EVAL_DIR = 'eval_results/fusion_comparison'
os.makedirs(EVAL_DIR, exist_ok=True)
save_data = {}
for ood_pcap in ood_pcaps:
    proto = ood_pcap.replace('.pcap','')
    ood_idx = [i for i, p in enumerate(all_protocols) if p == ood_pcap]
    if not ood_idx: continue
    X_test = X[ood_idx]; y_test = y[ood_idx]
    proto_data = {}
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        proto_data[name] = compute_metrics(y_test, y_pred)
    if ood_pcap in gnn_by_source:
        proto_data['GNN4ID (raw)'] = {'recall': gnn_by_source[ood_pcap]['gnn_raw']['recall'], 'f1': gnn_by_source[ood_pcap]['gnn_raw']['f1']}
        proto_data['FieldGraph-Net (Fusion)'] = gnn_by_source[ood_pcap]['fusion_alert']
    save_data[proto] = proto_data
with open(os.path.join(EVAL_DIR, 'per_protocol_baselines.json'), 'w') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)
print('\nSaved to eval_results/fusion_comparison/per_protocol_baselines.json')
print('\nDone.')
