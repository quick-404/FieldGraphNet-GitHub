# FieldGraph-Net research artifact

This standalone export contains the pipeline, experimental scripts, model checkpoints, selected aggregate results, example PCAPs, tests, and the manuscript. The working tree and intermediate files are not included.

## Install

Use Python 3.10 and create an environment from the repository root.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt ".[test]"
python -m pytest tests -q
```

Ubuntu/Debian:

```bash
sudo apt-get install -y libpcap-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt ".[test]"
python -m pytest tests -q
```

`nfstream` needs Npcap on Windows or libpcap on Linux; the Scapy feature backend can be used for the included smoke test. `nemere` is not bundled; when absent, the code uses its built-in segmentation implementation.

## Run

Windows PowerShell:

```powershell
$env:NEMESYS_FEATURE_BACKEND='scapy'
python examples/create_sample.py
python -m nemesys_gnn4id analyze data/data/sample_http.pcap --max-flows 2 --device cpu --proto-model models/field_proto_model_v2.pth
```

On Unix, run `python examples/create_sample.py`, then `NEMESYS_FEATURE_BACKEND=scapy python -m nemesys_gnn4id analyze data/data/sample_http.pcap --max-flows 2 --device cpu --proto-model models/field_proto_model_v2.pth`.

`models/model.pth` is the default attack classifier. The root `field_proto_model.pth` is a byte-identical copy of `models/field_proto_model_v2.pth`, provided because the unchanged CLI defaults to that path and the older root checkpoint is incompatible with the current feature dimensions. Other checkpoints in `models/` are retained for provenance, not guaranteed interchangeability.

The included PCAP is generated locally using documentation-range IP addresses; it is only a smoke-test sample, not part of the CIC evaluation. `data/data/README.md` records the sources for the excluded protocol training traces.

## Experimental inputs

Run scripts from the repository root. The CIC IoT 2023 comparison and ablation require separately obtained captures and their labels under `data/data/23pcap_chunks/` (including `_chunk_index.json`). The CICIDS2017 time-window analysis requires the corresponding Wednesday and Friday captures under `data/data/17pcap_chunks/`. The scripts under `experiments/` contain the exact paths, arguments, and analysis logic; for example, run `python experiments/run_ablation_fusion.py --help`, `python experiments/ablation_controlled_fpr.py --help`, or `python experiments/cicids2017_timewindow_eval.py --help`.

Raw CIC datasets and per-flow traces are **not included**. Selected aggregate JSON reports are retained in `eval_results/fusion_comparison/`; they do not substitute for underlying captures or missing per-flow intermediates. The example analysis can be run immediately; full paper experiments require acquiring the datasets and regenerating their intermediates. The manuscript's same-set controlled-FPR points should not be interpreted as independently validated deployment performance.

## Manuscript

`paper/lunwenV2.tex` preserves the requested rollback state. Build from `paper/` with `pdflatex lunwenV2.tex`, `bibtex lunwenV2`, then `pdflatex lunwenV2.tex` twice. The Open Science section links to the anonymized artifact. The supplied PDF is for inspection, not a camera-ready compliance claim.

No software license is granted by this export. Obtain permission from the rights holders before adding an explicit license; third-party dataset and model terms are separate.
