"""E7 G2 read-only CIC-IDS2017 source padding and timestamp-order forensics.

Output is aggregate-only. Never exports source IPs, network endpoints or row data;
never modifies original bytes, fixes timestamps, imputes labels or trains a model.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical(s):
    return ''.join(c.lower() for c in str(s) if c.isalnum())

def checked(path,encoding):
    p=Path(path)
    if not p.is_file() or p.suffix.lower()!='.csv' or 'unsw' in p.name.lower():
        raise ValueError('Only original external source CSV permitted')
    if encoding not in ('utf-8-sig','cp1252'):
        raise ValueError('Encoding must come from independently verified G0 evidence')
    return p

def missing_block(path,*,encoding,chunk_size=20000):
    p=checked(path,encoding)
    n=0;fields=None;counts=Counter();hist=Counter()
    first_missing=last_missing=first_null=last_null=first_labeled_no_time=None
    missing_runs=null_runs=0;active_missing=active_null=0
    longest_missing=longest_null=0;trailing_missing=trailing_null=0
    prior_missing=prior_null=False
    for chunk in pd.read_csv(p,encoding=encoding,dtype='string',chunksize=chunk_size,
                             low_memory=False,on_bad_lines='error'):
        mapping={canonical(k):k for k in chunk.columns}
        if fields is None:
            required=('label','timestamp','sourceip','destinationip','flowduration')
            if any(k not in mapping for k in required):
                raise ValueError('Required original source columns missing')
            fields=len(chunk.columns)
        label=chunk[mapping['label']]
        stamp=chunk[mapping['timestamp']]
        ml=(label.isna()|label.fillna('').str.strip().eq('')).to_numpy(dtype=bool)
        mt=(stamp.isna()|stamp.fillna('').str.strip().eq('')).to_numpy(dtype=bool)
        nonempty=np.zeros(len(chunk),dtype=np.int32)
        for col in chunk.columns:
            nonempty+=chunk[col].fillna('').str.strip().ne('').to_numpy(dtype=np.int32)
        blank=nonempty==0
        hist.update({str(int(k)):int(v) for k,v in zip(*np.unique(nonempty,return_counts=True))})
        counts['no_label']+=int(ml.sum())
        counts['no_timestamp']+=int(mt.sum())
        counts['all_fields_empty']+=int(blank.sum())
        counts['missing_label_and_missing_timestamp']+=int(np.count_nonzero(ml&mt))
        counts['missing_label_but_other_fields_present']+=int(np.count_nonzero(ml&~blank))
        counts['labeled_without_timestamp']+=int(np.count_nonzero(~ml&mt))
        counts['missing_label_and_all_fields_empty']+=int(np.count_nonzero(ml&blank))
        counts['label_present_and_fields_all_empty']+=int(np.count_nonzero(~ml&blank))
        for m,e,t in zip(ml,blank,mt):
            n+=1
            if m:
                if first_missing is None:first_missing=n
                last_missing=n
                active_missing+=1
                if not prior_missing:missing_runs+=1
                longest_missing=max(longest_missing,active_missing)
                trailing_missing+=1
            else:active_missing=trailing_missing=0
            prior_missing=bool(m)
            if e:
                if first_null is None:first_null=n
                last_null=n
                active_null+=1
                if not prior_null:null_runs+=1
                longest_null=max(longest_null,active_null)
                trailing_null+=1
            else:active_null=trailing_null=0
            prior_null=bool(e)
            if not m and t and first_labeled_no_time is None:first_labeled_no_time=n
    if not n:raise ValueError('Empty source CSV')
    return {
      'schema':'E7_G2_MISSINGNESS_FORENSICS_001',
      'source_file':p.name,'source_sha256':sha256_file(p),'source_encoding':encoding,
      'rows':n,'source_header_columns':fields,
      'counts':dict(sorted(counts.items())),
      'nonempty_fields_per_source_row_histogram':dict(sorted(hist.items(),key=lambda z:int(z[0]))),
      'first_missing_label_row_1based':first_missing,
      'last_missing_label_row_1based':last_missing,
      'missing_label_runs_in_csv_order':missing_runs,
      'longest_missing_label_run':longest_missing,
      'trailing_missing_label_rows':trailing_missing,
      'first_all_fields_empty_row_1based':first_null,
      'last_all_fields_empty_row_1based':last_null,
      'all_fields_empty_runs_in_csv_order':null_runs,
      'longest_all_fields_empty_run':longest_null,
      'trailing_all_fields_empty_rows':trailing_null,
      'first_labeled_without_timestamp_row_1based':first_labeled_no_time,
      'training_authorized':False,'source_modified':False,
      'interpretation':'Aggregate structural audit only. A blank block may mean padding or damaged CSV; no source-causation claim.'
    }

def clock_order(path,*,encoding,timestamp_format,expected_weekday,chunk_size=40000):
    p=checked(path,encoding)
    prev=None;rows=bad=reverse=tie=forward=wrap=other=0
    buckets=Counter();hours=Counter();weekdays=Counter()
    for chunk in pd.read_csv(p,encoding=encoding,dtype='string',chunksize=chunk_size,
                             usecols=lambda c:canonical(c)=='timestamp',on_bad_lines='error'):
        if chunk.shape[1]!=1:raise ValueError('Source Timestamp header ambiguous or absent')
        ts=pd.to_datetime(chunk.iloc[:,0],format=timestamp_format,errors='coerce',exact=True)
        rows+=len(ts);bad+=int(ts.isna().sum())
        valid=ts.dropna()
        hours.update({str(int(h)).zfill(2):int(v) for h,v in valid.dt.hour.value_counts().items()})
        weekdays.update({str(h):int(v) for h,v in valid.dt.day_name().value_counts().items()})
        for t in valid:
            if prev is not None:
                secs=(t-prev).total_seconds()
                if secs<0:
                    reverse+=1
                    if prev.hour==12 and t.hour==1:wrap+=1
                    else:other+=1
                    if secs>=-60:buckets['reversal_within_60sec']+=1
                    elif secs>=-3600:buckets['reversal_within_1hour']+=1
                    else:buckets['reversal_more_than_1hour']+=1
                elif secs==0:tie+=1
                else:forward+=1
            prev=t
    if not rows:raise ValueError('Empty source CSV')
    return {
      'schema':'E7_G2_TIMESTAMP_ORDER_FORENSICS_001',
      'source_file':p.name,'source_sha256':sha256_file(p),
      'source_encoding':encoding,'declared_time_format':timestamp_format,
      'expected_weekday':expected_weekday,'rows':rows,'parse_failures':bad,
      'raw_hour_distribution':dict(sorted(hours.items())),
      'parsed_weekday_distribution':dict(sorted(weekdays.items())),
      'adjacent_reversals_total':reverse,
      'adjacent_12_to_1_hour_reversals':wrap,
      'adjacent_other_reversals':other,
      'adjacent_tied_timestamps':tie,'adjacent_forward_steps':forward,
      'adjacent_reversal_magnitude_buckets':dict(sorted(buckets.items())),
      'weekday_consistent':len(weekdays)==1 and expected_weekday in weekdays and bad==0,
      'source_modified':False,'clock_correction_applied':False,
      'causal_order_verified':False,'training_authorized':False,
      'interpretation':'12-to-1 reversals suggest AM/PM ambiguity, not proof of afternoon; other reversals exclude monotonic source-row time.'
    }
