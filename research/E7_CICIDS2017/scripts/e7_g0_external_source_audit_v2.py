"""E7 G0: read-only external IDS source provenance/schema/time audit.

Not an experiment. Cannot certify event chronology, flow causality, label truth,
leakage-free splitting, or out-of-domain model generalisation. No raw IP values
or flow identifiers are written to public JSON. Supports plain CSV and CSV ZIP.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

ALIASES = {
    'time': ('timestamp', 'ts', 'starttime', 'flowstarttime', 'flowtimestamp'),
    'src': ('sourceip', 'srcip', 'idorig_h', 'sourceaddress'),
    'dst': ('destinationip', 'dstip', 'idresp_h', 'destinationaddress'),
    'sport': ('sourceport', 'srcport', 'idorig_p'),
    'dport': ('destinationport', 'dstport', 'idresp_p'),
    'protocol': ('protocol', 'proto'),
    'label': ('label', 'attackcat', 'attackcategory', 'type'),
}

def canonical(name):
    return ''.join(c.lower() for c in str(name).lstrip('\ufeff') if c.isalnum() or c == '_').replace('_', '')

def find_columns(headers):
    norm = {}
    for h in headers:
        norm.setdefault(canonical(h), []).append(h)
    resolved = {}
    for key, synonyms in ALIASES.items():
        candidates = [norm[canonical(c)][0] for c in synonyms if canonical(c) in norm]
        if len(set(candidates)) > 1:
            raise ValueError(f'Ambiguous {key} columns: {candidates}')
        if candidates:
            resolved[key] = candidates[0]
    return resolved

def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def sources(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if 'UNSW' in path.name.upper() or 'TESTING-SET' in path.name.upper():
        raise ValueError('G0 accepts only selected EXTERNAL source, never frozen UNSW test')
    if path.suffix.lower() == '.csv':
        with path.open('rb') as handle:
            yield path.name, handle
    elif path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            members = [i for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith('.csv')]
            if not members:
                raise ValueError('No CSV files found in archive')
            for member in members:
                with archive.open(member, 'r') as handle:
                    yield member.filename, handle
    else:
        raise ValueError('Expected one CSV or a ZIP containing CSV files')

def inspect(path, *, timestamp_format, max_rows=0, chunk_size=50000, source_encoding='utf-8-sig'):
    if not timestamp_format:
        raise ValueError('Explicit --timestamp-format required; automatic inference is prohibited')
    summary = {
        'phase': 'E7_G0_READ_ONLY_DATA_PROVENANCE',
        'external_source_basename': Path(path).name,
        'input_artifact_sha256': fingerprint(path),
        'timestamp_format_declared': timestamp_format,
        'full_file_scan': max_rows == 0,
        'not_trained': True,
        'unsw_official_test_touched': False,
        'temporal_causality_verified': False,
        'no_raw_endpoint_identifiers_exported': True,
        'files': [],
    }
    required = set(ALIASES)
    for name, handle in sources(path):
        member = {'relative_csv_member': name, 'rows_inspected': 0, 'labels': {},
                  'label_day_counts': {}, 'parse_failures': 0,
                  'timestamp_disorder_adjacent': 0, 'missing_endpoint_rows': 0,
                  'status': 'SCANNING'}
        labels = collections.Counter()
        byday = collections.Counter()
        previous = None
        reader = pd.read_csv(handle, chunksize=chunk_size, low_memory=False, encoding=source_encoding, on_bad_lines='error')
        resolved = None
        for chunk in reader:
            if resolved is None:
                member['original_column_names'] = list(chunk.columns)
                resolved = find_columns(chunk.columns)
                missing = sorted(required - set(resolved))
                member['columns_resolved'] = resolved
                if missing:
                    raise ValueError(f'{name}: missing required fields {missing}; do not substitute MachineLearningCSV for GeneratedLabelledFlows')
            if max_rows and member['rows_inspected'] >= max_rows:
                break
            if max_rows:
                chunk = chunk.iloc[:max_rows-member['rows_inspected']].copy()
            if len(chunk) == 0:
                break
            member['rows_inspected'] += len(chunk)
            label = chunk[resolved['label']].astype('string').fillna('__MISSING__').str.strip()
            labels.update(label.value_counts().to_dict())
            times = pd.to_datetime(chunk[resolved['time']], format=timestamp_format,
                                   errors='coerce', exact=True)
            member['parse_failures'] += int(times.isna().sum())
            valid = times.dropna()
            if len(valid):
                first, last = valid.iloc[0], valid.iloc[-1]
                if previous is not None and first < previous:
                    member['timestamp_disorder_adjacent'] += 1
                member['timestamp_disorder_adjacent'] += int((valid.diff().dt.total_seconds() < 0).sum())
                previous = last
                dates = times.dt.strftime('%Y-%m-%d').fillna('__BAD_TIME__')
                byday.update((date + ' | ' + lab) for date, lab in zip(dates.tolist(), label.tolist()))
            endpoint_cols = [resolved[x] for x in ('src','dst','sport','dport','protocol')]
            member['missing_endpoint_rows'] += int(chunk[endpoint_cols].isna().any(axis=1).sum())
        member['labels'] = dict(sorted((str(k), int(v)) for k,v in labels.items()))
        member['label_day_counts'] = dict(sorted((str(k), int(v)) for k,v in byday.items()))
        member['timestamp_parse_fraction'] = (
            1 - member['parse_failures']/member['rows_inspected'] if member['rows_inspected'] else None)
        member['status'] = ('SCHEMA_VALID_NOT_CAUSALITY_CERTIFIED' if member['rows_inspected']
                            and member['parse_failures'] == 0 and member['missing_endpoint_rows'] == 0
                            else 'DATA_QUALITY_REVIEW_REQUIRED')
        summary['files'].append(member)
    summary['total_rows_inspected'] = sum(x['rows_inspected'] for x in summary['files'])
    summary['status'] = ('G0_SCHEMA_CHECKED_TEMPORAL_VALIDITY_PENDING' if
        summary['full_file_scan'] and all(x['status']=='SCHEMA_VALID_NOT_CAUSALITY_CERTIFIED' for x in summary['files'])
        else 'G0_NOT_CLEARED')
    summary['explicit_followup_gates'] = [
        'Confirm publisher/corrected provenance and data license independently',
        'Verify attack timestamp timezone, start/end semantics and duration units',
        'Verify source/destination roles, bidirectional flow association and session IDs',
        'Verify corrected versus original label/feature version separately',
        'Inspect duplicate groups, unique sessions, overlap and capture-day attack schedules',
        'Preregister chronology-specific train/calibration/test splits before fitting',
        'No flow windows until all provenance and causal-end checks pass',
    ]
    if source_encoding != 'utf-8-sig':
        summary['explicit_followup_gates'].append(
            'ENCODING_REVIEW_REQUIRED: Original source UTF-8 failed strict validation; '
            'source was decoded reversibly using '+source_encoding+
            ' with no replacement. Validate original labels and text semantics independently.')
        for member in summary['files']:
            member['status'] = 'DATA_QUALITY_REVIEW_REQUIRED'
        summary['status'] = 'G0_NOT_CLEARED'
    return summary

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Exact existing external CSV or ZIP path')
    parser.add_argument('--output', required=True, help='Non-sensitive JSON output path')
    parser.add_argument('--timestamp-format', required=True,
                        help='Explicit pandas strptime format e.g. %%d/%%m/%%Y %%H:%%M:%%S')
    parser.add_argument('--max-rows', type=int, default=0,
                        help='0=full scan; positive number per CSV is SMOKE ONLY and cannot clear G0')
    args = parser.parse_args(argv)
    report = inspect(args.data, timestamp_format=args.timestamp_format, max_rows=args.max_rows)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('phase','status','total_rows_inspected','input_artifact_sha256')}, indent=2))
    print('G0 only. No model training, time-causality claim or official UNSW test access.')

if __name__ == '__main__':
    main()
