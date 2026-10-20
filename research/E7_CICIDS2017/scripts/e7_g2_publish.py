"""Append-only E7 G2 GitHub publisher; safe aggregates and immutable receipts only."""
from __future__ import annotations
import base64,hashlib,json,re,time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request,urlopen

REPO='itsCodeBakery/rare-attack-spatiotemporal-ids'
PREFIX='research/E7_CICIDS2017/results/G2_SOURCE_FORENSICS'
ALLOWED={
 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
 'Monday-WorkingHours.pcap_ISCX.csv',
 'Tuesday-WorkingHours.pcap_ISCX.csv',
 'Wednesday-workingHours.pcap_ISCX.csv'
}
MISSING_KEYS=set(('schema source_file source_sha256 source_encoding rows source_header_columns counts '
 'nonempty_fields_per_source_row_histogram first_missing_label_row_1based '
 'last_missing_label_row_1based missing_label_runs_in_csv_order longest_missing_label_run '
 'trailing_missing_label_rows first_all_fields_empty_row_1based last_all_fields_empty_row_1based '
 'all_fields_empty_runs_in_csv_order longest_all_fields_empty_run trailing_all_fields_empty_rows '
 'first_labeled_without_timestamp_row_1based training_authorized source_modified interpretation').split())
CLOCK_KEYS=set(('schema source_file source_sha256 source_encoding declared_time_format '
 'expected_weekday rows parse_failures raw_hour_distribution parsed_weekday_distribution '
 'adjacent_reversals_total adjacent_12_to_1_hour_reversals adjacent_other_reversals '
 'adjacent_tied_timestamps adjacent_forward_steps adjacent_reversal_magnitude_buckets '
 'weekday_consistent source_modified clock_correction_applied causal_order_verified '
 'training_authorized interpretation').split())
COUNT_KEYS=set(('no_label no_timestamp all_fields_empty missing_label_and_missing_timestamp '
 'missing_label_but_other_fields_present labeled_without_timestamp '
 'missing_label_and_all_fields_empty label_present_and_fields_all_empty').split())

def validate(record):
    name=record.get('source_file')
    if name not in ALLOWED:raise ValueError('Unexpected G2 source filename')
    if not re.fullmatch('[a-f0-9]{64}',str(record.get('source_sha256'))):
        raise ValueError('Missing original source checksum')
    if not isinstance(record.get('rows'),int) or record['rows']<1:
        raise ValueError('Invalid row count')
    if record.get('training_authorized') is not False or record.get('source_modified') is not False:
        raise ValueError('Cannot publish trained or source-modified result')
    if record.get('schema')=='E7_G2_MISSINGNESS_FORENSICS_001':
        if set(record)!=MISSING_KEYS or name!='Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv':
            raise ValueError('Unsupported G2 missingness report schema')
        if set(record['counts'])-COUNT_KEYS or not all(isinstance(n,int) and n>=0 for n in record['counts'].values()):
            raise ValueError('Source quality count schema mismatch')
        if not all(str(k).isdigit() and isinstance(v,int) and v>=0
                   for k,v in record['nonempty_fields_per_source_row_histogram'].items()):
            raise ValueError('Invalid numeric row-density histogram')
    elif record.get('schema')=='E7_G2_TIMESTAMP_ORDER_FORENSICS_001':
        if set(record)!=CLOCK_KEYS or name.startswith('Thursday'):
            raise ValueError('Unexpected clock source or fields')
        if record.get('causal_order_verified') is not False or record.get('clock_correction_applied') is not False:
            raise ValueError('Unsafe temporal verification claim')
        if record['adjacent_12_to_1_hour_reversals']+record['adjacent_other_reversals']!=record['adjacent_reversals_total']:
            raise ValueError('Invalid reversal accounting')
        for k in ('raw_hour_distribution','parsed_weekday_distribution','adjacent_reversal_magnitude_buckets'):
            if not isinstance(record[k],dict):raise ValueError('Invalid hour aggregate')
    else:raise ValueError('Unrecognized G2 phase')
    return True

def canonical_bytes(report):
    validate(report)
    return (json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode('utf-8')

def api(method,url,token,body=None):
    if not token:raise ValueError('Missing GitHub write secret')
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
             'X-GitHub-Api-Version':'2022-11-28'}
    data=None if body is None else json.dumps(body).encode()
    if data is not None:headers['Content-Type']='application/json'
    with urlopen(Request(url,method=method,headers=headers,data=data),timeout=65) as resp:
        return json.load(resp)

def verify(url,token,source,blob=None,commit=None):
    links=[]
    if blob:links.append(('blob','https://api.github.com/repos/'+REPO+'/git/blobs/'+blob))
    if commit:links.append(('commit',url+'?ref='+quote(commit,safe='')))
    links.append(('current',url))
    for kind,link in links:
        try:response=api('GET',link,token)
        except HTTPError as err:
            if err.code in (404,409,429,500,502,503,504):continue
            raise
        remote=base64.b64decode(response['content'])
        if remote!=source:raise RuntimeError('GitHub stored conflicting report; NEVER overwrite')
        return kind
    return None

def write_receipt(target,obj):
    p=Path(target)
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.writing')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    tmp.replace(p)

def publish(report,*,token,local_dir,pause=time.sleep):
    data=canonical_bytes(report)
    folder=Path(local_dir)
    folder.mkdir(parents=True,exist_ok=True)
    filename=report['source_file']+'.g2.json'
    file=folder/filename
    if file.exists():
        if file.read_bytes()!=data:raise RuntimeError('Local checkpoint differs; refuse to overwrite')
    else:file.write_bytes(data)
    receipt=folder/(filename+'.receipt.json')
    remote=PREFIX+'/'+filename
    digest=hashlib.sha256(data).hexdigest()
    url='https://api.github.com/repos/'+REPO+'/contents/'+quote(remote,safe='/')
    if receipt.exists():
        prior=json.loads(receipt.read_text(encoding='utf-8'))
        if prior.get('sha256')!=digest or prior.get('remote')!=remote:
            raise RuntimeError('Saved receipt does not match source report')
        blob=prior.get('blob');commit=prior.get('commit')
    else:blob=commit=None
    for seconds in (0,1,2,3,5,8):
        if seconds:pause(seconds)
        state=verify(url,token,data,blob,commit)
        if state:
            write_receipt(receipt,{'remote':remote,'sha256':digest,
                                   'blob':blob,'commit':commit,'verified_via':state})
            return {'status':'VERIFIED','remote':remote,'via':state}
        if not receipt.exists():break
    if receipt.exists():
        raise RuntimeError('Previously accepted PUT may still be pending; no duplicate upload')
    try:
        response=api('PUT',url,token,{
            'message':'results(e7-g2): immutable source-forensics report '+report['source_file'],
            'branch':'main','content':base64.b64encode(data).decode('ascii')})
    except HTTPError as err:
        if err.code not in (409,422):raise
        for seconds in (0,1,2,3,5,8):
            if seconds:pause(seconds)
            if verify(url,token,data):
                write_receipt(receipt,{'remote':remote,'sha256':digest,
                                       'verified_via':'concurrent-existing'})
                return {'status':'VERIFIED_CONCURRENT','remote':remote}
        raise RuntimeError('Conflicting remote report exists') from err
    blob=response['content']['sha'];commit=response['commit']['sha']
    # Save returned immutable Git objects IMMEDIATELY before any verification calls.
    write_receipt(receipt,{'remote':remote,'sha256':digest,'blob':blob,'commit':commit,
                           'status':'PUT_ACCEPTED_VERIFY_PENDING'})
    for seconds in (0,1,2,3,5,8,13):
        if seconds:pause(seconds)
        state=verify(url,token,data,blob,commit)
        if state:
            write_receipt(receipt,{'remote':remote,'sha256':digest,'blob':blob,
                                   'commit':commit,'verified_via':state})
            return {'status':'UPLOADED_VERIFIED','remote':remote,'via':state}
    raise RuntimeError('PUT accepted, receipt kept, verification pending. Do not repeat PUT.')
