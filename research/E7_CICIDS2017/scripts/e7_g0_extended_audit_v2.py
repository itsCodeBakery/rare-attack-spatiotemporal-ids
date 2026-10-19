"""E7 versioned encoding-aware read-only audit, derived from original preserved module.

Original previous findings and source code retained without modification.
Version 2 adds explicit per-file source_encoding after strict byte validation;
never uses replacement decoding or silently alters original CSV source bytes.
"""
"""E7 full-source extended quality audit (no training or input modification).

Scans original CSV in chunks. Only aggregate outputs, NEVER source flows/IP
addresses or per-example hashes, are returned. A flow fingerprint used for
candidate duplicate counts is retained only in volatile Python memory.
It is hash-based and NOT an exact collision-free duplicate certificate.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

EXCLUDE = {'flowid','sourceip','srcip','destinationip','dstip','timestamp',
           'label','srcaddress','dstaddress'}
def norm(value):
    return ''.join(x for x in str(value).strip().lower() if x.isalnum())

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def audit_file(path, *, timestamp_format, expected_weekday, chunk_size=35000, source_encoding='utf-8-sig'):
    path=Path(path)
    if not path.is_file() or path.suffix.lower()!='.csv':
        raise ValueError('Expected one actual source CSV')
    if 'unsw' in path.name.lower() or 'testing-set' in path.name.lower():
        raise ValueError('Official UNSW test may not be accessed in E7')
    basename=path.name
    original_hash=sha256_file(path)
    labels=Counter()
    hours=Counter()
    weekday_counts=Counter()
    nulls=Counter()
    infinities=Counter()
    negatives=Counter()
    zeros=Counter()
    minima={}
    maxima={}
    count=0
    timestamp_bad=0
    disorder=0
    previous=None
    first_stamp=None
    last_stamp=None
    raw_hour_lt12=False
    duration_bad=duration_zero=duration_negative=duration_nonfinite=0
    duration_min=None
    duration_max=None
    duration_field=None
    numeric_cols=None
    missing_endpoints=0
    seen={}
    duplicates=0
    contradictions=0
    no_identifiers=('flowid','sourceip','destinationip','sourceport','destinationport')
    for chunk in pd.read_csv(path,dtype='string',
                             chunksize=chunk_size,on_bad_lines='error',low_memory=False,encoding=source_encoding):
        if chunk.empty: continue
        if numeric_cols is None:
            mapping={norm(c):c for c in chunk.columns}
            for req in ('timestamp','label','sourceip','destinationip',
                        'sourceport','destinationport','protocol'):
                if req not in mapping:
                    raise ValueError(basename+': missing required original flow field '+req)
            timecol=mapping['timestamp']
            labelcol=mapping['label']
            duration_field=mapping.get('flowduration')
            if duration_field is None:
                raise ValueError(basename+': original flow duration field missing')
            endpoints=[mapping[c] for c in
                       ('sourceip','destinationip','sourceport','destinationport','protocol')]
            numeric_cols=[c for c in chunk.columns if norm(c) not in EXCLUDE]
            if len(set(chunk.columns))!=len(chunk.columns):
                raise ValueError('Duplicate exact source column names')
        count+=len(chunk)
        y=chunk[labelcol].astype('string').fillna('__MISSING_LABEL__').str.strip()
        labels.update({str(k):int(v) for k,v in y.value_counts().items()})
        ts=pd.to_datetime(chunk[timecol],format=timestamp_format,
                          errors='coerce',exact=True)
        timestamp_bad+=int(ts.isna().sum())
        good=ts.dropna()
        if len(good):
            if previous is not None and good.iloc[0]<previous:disorder+=1
            disorder+=int((good.diff().dt.total_seconds()<0).sum())
            previous=good.iloc[-1]
            lo=good.min()
            hi=good.max()
            first_stamp=lo if first_stamp is None else min(first_stamp,lo)
            last_stamp=hi if last_stamp is None else max(last_stamp,hi)
            weekday_counts.update(good.dt.day_name().value_counts().to_dict())
            hours.update({str(k):int(v) for k,v in good.dt.hour.value_counts().items()})
            if 'afternoon' in basename.lower() and bool((good.dt.hour<12).any()):
                raw_hour_lt12=True
        missing_endpoints+=int(chunk[endpoints].isna().any(axis=1).sum())
        # Source precision remains literal string. No automatic AM/PM correction.
        dur=pd.to_numeric(chunk[duration_field],errors='coerce')
        dvals=dur.to_numpy(dtype=np.float64,na_value=np.nan)
        finite=np.isfinite(dvals)
        duration_bad+=int(np.isnan(dvals).sum())
        duration_nonfinite+=int(np.isinf(dvals).sum())
        duration_negative+=int((dvals[finite]<0).sum())
        duration_zero+=int((dvals[finite]==0).sum())
        if finite.any():
            dmin=float(dvals[finite].min())
            dmax=float(dvals[finite].max())
            duration_min=dmin if duration_min is None else min(duration_min,dmin)
            duration_max=dmax if duration_max is None else max(duration_max,dmax)
        for col in numeric_cols:
            data=pd.to_numeric(chunk[col],errors='coerce')
            values=data.to_numpy(dtype=np.float64,na_value=np.nan)
            finite_col=np.isfinite(values)
            nulls[col]+=int(np.isnan(values).sum())
            infinities[col]+=int(np.isinf(values).sum())
            negatives[col]+=int((values[finite_col]<0).sum())
            zeros[col]+=int((values[finite_col]==0).sum())
            if finite_col.any():
                curmin=float(values[finite_col].min())
                curmax=float(values[finite_col].max())
                minima[col]=curmin if col not in minima else min(minima[col],curmin)
                maxima[col]=curmax if col not in maxima else max(maxima[col],curmax)
        # Candidate duplicate: all raw columns except label, including raw time
        # and endpoints. Fingerprints never leave process or public report.
        fingerprints=pd.util.hash_pandas_object(
            chunk.drop(columns=[labelcol]),index=False).to_numpy(dtype=np.uint64)
        for hashid,lab in zip(fingerprints,y):
            key=int(hashid)
            original=seen.get(key)
            if original is None:
                seen[key]=str(lab)
            else:
                duplicates+=1
                if original!=str(lab):
                    contradictions+=1
    if not count:raise ValueError(basename+': empty CSV')
    constant=[col for col in numeric_cols if col in minima
              and minima[col]==maxima[col]]
    valid_weekday=(set(weekday_counts)=={expected_weekday} and not timestamp_bad)
    # Source columns/feature names are public; no IP/port/Flow ID values exported.
    report={
      'schema':'E7_G0_EXTENDED_AGGREGATE_001','status':'REVIEW_REQUIRED',
      'source_file':basename,'source_sha256':original_hash,'rows':int(count),
      'labels':dict(sorted(labels.items())),
      'timestamp_quality':{
        'raw_hour_histogram':dict(sorted(hours.items(),key=lambda kv:int(kv[0]))),
        'parsed_weekdays':dict(sorted(weekday_counts.items())),
        'parse_failures':int(timestamp_bad),
        'row_order_reversals':int(disorder),
        'timestamp_min':first_stamp.isoformat() if first_stamp is not None else None,
        'timestamp_max':last_stamp.isoformat() if last_stamp is not None else None,
        'clock_ambiguous':bool(raw_hour_lt12 or not valid_weekday)},
      'duration_quality':{
        'duration_unit':'UNVERIFIED_ORIGINAL_FLOW_DURATION_UNITS',
        'duration_zero':int(duration_zero),
        'duration_negative':int(duration_negative),
        'duration_missing':int(duration_bad),
        'duration_nonfinite':int(duration_nonfinite),
        'duration_max_raw':duration_max,'duration_min_raw':duration_min},
      'feature_quality':{
        'numeric_feature_count':len(numeric_cols),
        'columns_with_missing':dict(sorted((k,int(v)) for k,v in nulls.items() if v)),
        'columns_with_nonfinite':dict(sorted((k,int(v)) for k,v in infinities.items() if v)),
        'columns_with_negative':dict(sorted((k,int(v)) for k,v in negatives.items() if v)),
        'columns_with_zero':dict(sorted((k,int(v)) for k,v in zeros.items() if v)),
        'constant_numeric_columns':sorted(constant),
        'missing_endpoint_rows':int(missing_endpoints)},
      'duplicates':{
        'candidate_duplicate_rows':int(duplicates),
        'candidate_label_conflict_rows':int(contradictions),
        'hash_scheme':'pandas 64-bit row hash over original source fields excluding Label',
        'disclaimer':'Hash collisions possible. Fingerprints processed in memory only and never exported. Not proof of leakage.'},
      'notes':[
        'Quality and chronology diagnostics only. No model fitting or UNSW official test.',
        'No causal flow end/start or timestamp timezone proven. AM/PM for afternoon files unresolved.',
        'CICIDS2017 originals and corrected Engelen dataset must be independently evaluated.',
        'Source row order, file day and per-minute timestamps are not guaranteed related-flow chronology.',
        'Unseen day-specific attacks make day-holdout multiclass classification an open-world task.',
        'This report is always REVIEW_REQUIRED until independent chronology, label and source verification.',
        'Original CSV encoding: '+source_encoding+'. Non-UTF8 text must be independently reviewed; no byte replacement.']
    }
    return report
