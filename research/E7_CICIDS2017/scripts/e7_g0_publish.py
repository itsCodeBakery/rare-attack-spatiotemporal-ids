"""Fail-closed GitHub publisher for public, aggregate-only E7 G0 reports.

Original traffic, IP values, per-row predictions, tokens, and models are never uploaded.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = 'itsCodeBakery/rare-attack-spatiotemporal-ids'
PREFIX = 'research/E7_CICIDS2017/results/G0'

def verify_report(report):
    if report.get('phase') != 'E7_G0_READ_ONLY_DATA_PROVENANCE':
        raise ValueError('Not an original E7 G0 read-only audit')
    for k, v in [('not_trained', True), ('unsw_official_test_touched', False),
                 ('temporal_causality_verified', False),
                 ('no_raw_endpoint_identifiers_exported', True), ('full_file_scan', True)]:
        if report.get(k) is not v:
            raise ValueError('Failed report safeguard: ' + k)
    if report.get('status') not in ('G0_SCHEMA_CHECKED_TEMPORAL_VALIDITY_PENDING','G0_NOT_CLEARED'):
        raise ValueError('Unrecognized G0 audit result status')
    files = report.get('files')
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError('Expected one audited input CSV per report')
    item = files[0]
    for x in ('src','dst','sport','dport','protocol','time','label'):
        if x not in item.get('columns_resolved', {}):
            raise ValueError('Missing required source schema: ' + x)
    if item.get('status') not in ('SCHEMA_VALID_NOT_CAUSALITY_CERTIFIED','DATA_QUALITY_REVIEW_REQUIRED'):
        raise ValueError('Unrecognized per-file audit status')
    if not item.get('rows_inspected'):
        raise ValueError('No rows inspected')
    good = item['status']=='SCHEMA_VALID_NOT_CAUSALITY_CERTIFIED'
    if good != (report['status']=='G0_SCHEMA_CHECKED_TEMPORAL_VALIDITY_PENDING'):
        raise ValueError('Inconsistent checked versus failed G0 status')
    if not item.get('relative_csv_member', '').endswith('.csv'):
        raise ValueError('Invalid CSV member')
    if not report.get('input_artifact_sha256') or len(report['input_artifact_sha256']) != 64:
        raise ValueError('Missing source SHA256')
    # Strict schema limits accidental publication of raw flow-level content.
    allowed_top = {'phase','external_source_basename','input_artifact_sha256',
      'timestamp_format_declared','full_file_scan','not_trained','unsw_official_test_touched',
      'temporal_causality_verified','no_raw_endpoint_identifiers_exported','files',
      'total_rows_inspected','status','explicit_followup_gates'}
    if set(report) - allowed_top:
        raise ValueError('Unexpected report keys; manual review before GitHub upload')
    allowed_item = {'relative_csv_member','rows_inspected','labels','label_day_counts',
       'parse_failures','timestamp_disorder_adjacent','missing_endpoint_rows',
       'status','original_column_names','columns_resolved','timestamp_parse_fraction'}
    if set(item) - allowed_item:
        raise ValueError('Unexpected per-file fields; manual review before GitHub upload')
    return True

def prepare_public_blob(filepath):
    filepath = Path(filepath)
    report = json.loads(filepath.read_text(encoding='utf-8'))
    verify_report(report)
    data = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')
    return data, report

def github_json(method, api_url, token, obj=None):
    if not token:
        raise ValueError('Missing GITHUB_TOKEN: halt before publishing')
    headers={'Authorization': 'Bearer '+token, 'Accept':'application/vnd.github+json',
             'X-GitHub-Api-Version':'2022-11-28'}
    data = None if obj is None else json.dumps(obj).encode('utf-8')
    if data is not None: headers['Content-Type']='application/json'
    request = Request(api_url, data=data, method=method, headers=headers)
    with urlopen(request, timeout=65) as response:
        return json.load(response)

def publish_report(local_report, *, token, run_path):
    data, report = prepare_public_blob(local_report)
    if Path(run_path).name != 'G0' or not str(run_path).startswith(PREFIX):
        raise ValueError('Only designated E7 G0 path may be published')
    target_name=report['external_source_basename'] + '.audit.json'
    if '/' in target_name or '\\' in target_name or '..' in target_name:
        raise ValueError('Unsafe input filename')
    remote=f'{PREFIX}/{target_name}'
    url='https://api.github.com/repos/'+REPO+'/contents/'+quote(remote,safe='/')
    try:
        present=github_json('GET',url,token)
    except HTTPError as e:
        if e.code != 404: raise
        present=None
    if present is not None:
        decoded=base64.b64decode(present['content'])
        if decoded != data:
            raise RuntimeError('Existing E7 audit differs: refusing to overwrite original evidence')
        return {'status':'ALREADY_SAVED_IDENTICALLY','remote':remote,'sha256':hashlib.sha256(data).hexdigest()}
    message=f'results(e7-g0): original hashed source audit {report["external_source_basename"]}'
    response=github_json('PUT',url,token,{'message':message,
                    'content':base64.b64encode(data).decode('ascii'),'branch':'main'})
    remote_content=github_json('GET',url,token)
    if base64.b64decode(remote_content['content']) != data:
        raise RuntimeError('GitHub remote verification failed')
    return {'status':'UPLOADED_AND_VERIFIED','remote':remote,'commit':response['commit']['sha'],
            'sha256':hashlib.sha256(data).hexdigest()}
