import re

pdf = r'D:\网络流量分析项目\nemesys-gnn4id\【9】Scalable hierarchical protocol format inference via feature-heuristic message delimiter.pdf'

with open(pdf, 'rb') as f:
    raw = f.read()

text = raw.decode('latin-1')

# Find all URLs
urls = re.findall(r'https?://[^\s\)\>\"\(\)\[\]]+', text)
seen = set()

print("=== Dataset/Data related URLs ===")
for u in urls:
    cleaned = u.replace('\\', '')
    if any(kw in cleaned.lower() for kw in ['github', 'dataset', 'pcap', 'trace', 'download', 'data', 'doi', 'ctu', 'ics']):
        if cleaned not in seen:
            seen.add(cleaned)
            print(cleaned)

print("\n=== All URLs found in paper ===")
for u in urls[:40]:
    cleaned = u.replace('\\', '').rstrip(')').strip()
    print(cleaned[:250])
