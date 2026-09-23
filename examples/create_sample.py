"""Generate an artificial HTTP-like capture for a CLI smoke test."""

from pathlib import Path

from scapy.all import Ether, IP, TCP, Raw, wrpcap


def main():
    output = Path('data/data/sample_http.pcap')
    output.parent.mkdir(parents=True, exist_ok=True)
    packets = []
    for i in range(20):
        packets.append(
            Ether()
            / IP(src='192.0.2.1', dst='198.51.100.1')
            / TCP(sport=40000 + i, dport=80, flags='PA', seq=1)
            / Raw(load=f'GET /sample/{i} HTTP/1.1\r\nHost: example.test\r\n\r\n'.encode())
        )
    wrpcap(str(output), packets)
    print(output)


if __name__ == '__main__':
    main()
