"""E7 G1: source-label provenance and chronological feasibility, aggregates only.

Read-only. Does not infer original row ordering as causality, repair labels,
reinterpret afternoon clock, or expose raw endpoints. No training.
"""
from collections import Counter
from pathlib import Path
import hashlib
import pandas as pd

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def inspect(path, *, timestamp_format, expected_day, encoding="utf-8-sig", chunksize=35000):
    p=Path(path)
    if not p.is_file() or p.suffix.lower()!=".csv" or "unsw" in p.name.lower():
        raise ValueError("Only external CIC source CSV accepted")
    n=0; missing=0; runs=0; longest=0; inrun=0; previous_missing=False
    labels=Counter(); day_hour=Counter(); hour_missing=Counter()
    missing_start_row=None; first_missing=None; last_missing=None
    maxgap_seconds=None; reverse=0; ties=0; prior=None
    bad_ts=0; hours=Counter(); dates=Counter()
    original_sha=digest(p)
    for chunk in pd.read_csv(p,encoding=encoding,dtype="string",chunksize=chunksize,on_bad_lines="error"):
        mapping={''.join(ch.lower() for ch in str(col) if ch.isalnum()):col for col in chunk.columns}
        if "label" not in mapping or "timestamp" not in mapping:
            raise ValueError("Missing label or Timestamp column")
        raw=chunk[mapping["label"]].astype("string").str.strip()
        miss=raw.isna() | raw.eq("").fillna(False)
        normalized=raw.fillna("__MISSING_LABEL__").replace("", "__MISSING_LABEL__")
        labels.update({str(k):int(v) for k,v in normalized.value_counts(dropna=False).items()})
        timestamps=pd.to_datetime(chunk[mapping["timestamp"]],format=timestamp_format,
                                  exact=True,errors="coerce")
        for y,t,absent in zip(normalized,timestamps,miss):
            n+=1
            m=bool(absent)
            if m:
                missing+=1
                if not previous_missing:runs+=1;inrun=0
                inrun+=1
                longest=max(longest,inrun)
                if first_missing is None:first_missing=n
                last_missing=n
            else:inrun=0
            previous_missing=m
            if pd.isna(t):
                bad_ts+=1
                continue
            day=t.strftime("%Y-%m-%d")
            dates[day]+=1
            hour=int(t.hour)
            hours[str(hour).zfill(2)]+=1
            day_hour[day+" "+str(hour).zfill(2)+":00"+" | "+str(y)]+=1
            if m:hour_missing[day+" "+str(hour).zfill(2)+":00"]+=1
            if prior is not None:
                dt=(t-prior).total_seconds()
                if dt<0:reverse+=1
                if dt==0:ties+=1
                if dt>0:maxgap_seconds=dt if maxgap_seconds is None else max(maxgap_seconds,dt)
            prior=t
    assert n>0
    expected_weekdays={pd.Timestamp(k).day_name() for k in dates}
    return {
      "schema":"E7_G1_LABEL_CLOCK_DIAGNOSTIC_001",
      "source_file":p.name,"original_source_sha256":original_sha,
      "source_text_encoding":encoding,"rows":int(n),
      "missing_label_rows":int(missing),"missing_label_runs_in_csv_row_order":int(runs),
      "longest_missing_label_run_in_csv_row_order":int(longest),
      "first_missing_row_1based":first_missing,"last_missing_row_1based":last_missing,
      "labels":dict(sorted(labels.items())),
      "daily_row_counts":dict(sorted(dates.items())),
      "label_by_day_hour_counts":dict(sorted(day_hour.items())),
      "missing_labels_by_day_hour":dict(sorted(hour_missing.items())),
      "timestamp_hour_histogram":dict(sorted(hours.items())),
      "timestamp_parse_failures":int(bad_ts),
      "source_adjacent_ties":int(ties),
      "source_adjacent_time_reversals":int(reverse),
      "source_positive_max_gap_seconds":maxgap_seconds,
      "calendar_weekday_consistency":bool(expected_weekdays=={expected_day} and bad_ts==0),
      "causal_order_proven":False,"trusted_original_labels":False,
      "training_authorized":False,
      "review_flags":[
        "Original source labels not independently validated",
        "Source CSV row adjacency and filename-day agreement do not prove causal temporal history",
        "Afternoon clock convention may be ambiguous even if parsed timestamps have 0 errors",
        "Missing labels MUST NOT be converted to BENIGN, discarded silently or imputed",
        "Any study on corrected CICIDS2017 requires separate provenance/schema/label audit"
      ] + (["CRITICAL: ORIGINAL_SOURCE_MISSING_LABELS"] if missing else [])
          + (["CLOCK_AMPM_SOURCE_AMBIGUITY"] if "Afternoon" in p.name else [])
          + (["ENCODING_CP1252_NEEDS_SOURCE_REVIEW"] if encoding=="cp1252" else [])
    }
