# CIC-IDS2017 external data provenance

**State: source files discovered in Kaggle; original ZIP's publishing provenance, extracted CSV hashes and flow-time semantics not yet audited.** Do not claim data quality checks completed.

Source publisher: [UNB CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html). Required subsequent correction sensitivity: [Engelen et al. CIC-IDS2017 corrected labeled flows](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html).

Researcher-provided mount prefix: `/kaggle/input/datasets/dsshayan/dscicids/`. Kaggle printed eight CSV members under a directory resembling `TrafficLabelling ` (with a trailing space, immediately before `/`). **The originally typed path omitted that trailing space.** Discover the exact parent from the filesystem rather than trimming paths:

```python
from pathlib import Path
root = Path('/kaggle/input/datasets/dsshayan/dscicids')
files = sorted(root.rglob('*.csv'))
for p in files:
    print(repr(str(p.parent)), p.name, p.stat().st_size)
```

Expected user-observed basename list (not independently audited yet):

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

**Next required manifest (not yet available):** actual path representation; file size/hash; headers/time examples; parse formats and timezone; numeric duration unit and start/end; host and flow keys; attack/day class support; duplicate/conflicting-label checks; corrected-dataset provenance and mapping.

No raw flow rows/IP addresses committed. The existing G0 script performs **part** of this list and explicitly cannot certify causal flow relationships on its own. Do not train until missing gates are reviewed.
