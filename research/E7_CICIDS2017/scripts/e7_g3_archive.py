"""E7 G3 fail-closed model backup and safe public result publisher.

Model text goes ONLY to an unpublished GitHub draft release, never public git.
Every model package must pass end-to-end SHA256 read-back BEFORE held-out test
evaluation. Safe aggregate metrics only go to the regular GitHub research tree.
No tokens, raw flow records, predictions, endpoints or private URLs in public files.
"""
from __future__ import annotations
import base64,hashlib,io,json,time,re,zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request,urlopen

REPO='itsCodeBakery/rare-attack-spatiotemporal-ids'
TAG='e7-g3-openworld-dayholdout-001-private-models'
PUBLIC_PREFIX='research/E7_CICIDS2017/results/G3_OPENWORLD_DAYHOLDOUT'
ROOT='https://api.github.com/repos/'+REPO
MODEL_LIMIT=30_000_000

def sha(data):
    return hashlib.sha256(data).hexdigest()

def http(method,url,token,data=None,headers=None,parse_json=True):
    if not token:raise ValueError('Missing GitHub token; halt before model fitting')
    h={'Authorization':'Bearer '+token,'User-Agent':'E7-G3-Research',
       'X-GitHub-Api-Version':'2022-11-28',
       'Accept':'application/vnd.github+json'}
    if headers:h.update(headers)
    with urlopen(Request(url,data=data,headers=h,method=method),timeout=120) as response:
        raw=response.read()
        return json.loads(raw) if parse_json else raw

def api(method,url,token,payload=None):
    body=None if payload is None else json.dumps(payload).encode('utf-8')
    return http(method,url,token,body,
        {'Content-Type':'application/json'} if body is not None else None)

def draft_release(token):
    # Called in preflight BEFORE any expensive original-source model fitting.
    url=ROOT+'/releases/tags/'+quote(TAG,safe='')
    try:r=api('GET',url,token)
    except HTTPError as e:
        if e.code!=404:raise
        r=None
    if r is None:
        # Draft tags can be omitted by the release-by-tag API: check private list.
        listings=api('GET',ROOT+'/releases?per_page=100',token)
        matches=[item for item in listings if item.get('tag_name')==TAG]
        if len(matches)>1:
            raise RuntimeError('Multiple draft releases use locked G3 tag; stop')
        if matches:r=matches[0]
    if r is None:
        r=api('POST',ROOT+'/releases',token,{
            'tag_name':TAG,'target_commitish':'main',
            'name':'PRIVATE E7 G3 open-world original-source model checkpoints',
            'body':'Unpublished private models, not research conclusions. Do NOT publish.',
            'draft':True,'prerelease':True})
    if r.get('draft') is not True or r.get('tag_name')!=TAG:
        raise RuntimeError('Release is NOT a private draft; refuse model upload')
    if not r.get('upload_url') or not r.get('id'):
        raise RuntimeError('GitHub draft asset upload permission unavailable')
    # Verify credential access to draft assets before training.
    api('GET',ROOT+'/releases/'+str(r['id'])+'/assets?per_page=100',token)
    return r

def assets(release,token):
    assert release.get('draft') is True
    a=api('GET',ROOT+'/releases/'+str(release['id'])+'/assets?per_page=100',token)
    if not isinstance(a,list):raise RuntimeError('Could not list private draft release assets')
    return {x['name']:x for x in a}

def get_private_asset(asset,token):
    address=ROOT+'/releases/assets/'+str(asset['id'])
    raw=http('GET',address,token,headers={'Accept':'application/octet-stream'},parse_json=False)
    # Fail closed if GitHub API returns metadata JSON instead of asset binary.
    if raw.startswith(b'{') and b'browser_download_url' in raw[:4096]:
        meta=json.loads(raw)
        destination=meta.get('browser_download_url','')
        # Only follow the exact authenticated, owner-scoped release download URL.
        if not destination.startswith('https://github.com/'+REPO+'/releases/download/'):
            raise RuntimeError('Unexpected private draft download domain or repository')
        raw=http('GET',destination,token,
                 headers={'Accept':'application/octet-stream'},parse_json=False)
        if raw.startswith(b'{'):
            raise RuntimeError('GitHub private draft download returned metadata, not ZIP bytes')
    if not raw.startswith(b'PK\x03\x04'):
        raise RuntimeError('Draft release download did not return model ZIP bytes')
    if len(raw)>MODEL_LIMIT:
        raise RuntimeError('Oversized private model artifact')
    if asset.get('size')!=len(raw):
        raise RuntimeError('Private checkpoint download length differs')
    remote_digest=asset.get('digest')
    if remote_digest and remote_digest.startswith('sha256:'):
        if sha(raw)!=remote_digest.split(':',1)[1]:
            raise RuntimeError('Private GitHub model SHA256 does not match')
    return raw

def pack_model(booster,meta):
    assert meta['schema']=='E7_G3_PRIVATE_MODEL_001'
    assert isinstance(meta['feature_names'],list) and len(meta['feature_names'])>60
    data=booster.model_to_string().encode('utf-8')
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        archive.writestr('model.txt',data)
        archive.writestr('metadata.json',json.dumps(meta,sort_keys=True,indent=2)+'\n')
    raw=out.getvalue()
    assert len(raw)<=MODEL_LIMIT
    return raw

def unpack_model(raw,required):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if set(archive.namelist())!={'model.txt','metadata.json'}:
            raise ValueError('Unexpected files in model checkpoint')
        metadata=json.loads(archive.read('metadata.json'))
        for k,v in required.items():
            if metadata.get(k)!=v:raise RuntimeError('Private restored model metadata differs: '+k)
        model=archive.read('model.txt').decode('utf-8')
    import lightgbm as lgb
    return lgb.Booster(model_str=model),metadata

def archive_model(raw,*,token,release,asset_name,receipt_dir):
    assert release.get('draft') is True
    if asset_name!='G3_BACKUP_PREFLIGHT_PRIVATE.zip' and not re.fullmatch(r'G3_FOLD_[1-4]_SEED_4[2-6]_PRIVATE\.zip',asset_name):
        raise ValueError('Unexpected private G3 model asset filename')
    assert len(raw)<MODEL_LIMIT and raw.startswith(b'PK\x03\x04')
    folder=Path(receipt_dir);folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/(asset_name+'.receipt.json')
    current_sha=sha(raw)
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous.get('sha256')!=current_sha:
            raise RuntimeError('Original private archive receipt differs; no overwrite')
    found=assets(release,token).get(asset_name)
    if found:
        downloaded=get_private_asset(found,token)
        if downloaded!=raw:raise RuntimeError('Conflicting private model already on draft release')
        receipt.write_text(json.dumps({'asset':asset_name,'sha256':current_sha,
                           'bytes':len(raw),'verified':True},sort_keys=True)+'\n')
        return 'PRIVATE_EXISTING_VERIFIED'
    if receipt.exists():
        raise RuntimeError('Prior upload uncertain; no duplicate model upload')
    upload=release['upload_url'].split('{',1)[0]+'?name='+quote(asset_name)
    response=http('POST',upload,token,raw,
           {'Content-Type':'application/zip','Accept':'application/vnd.github+json'})
    if response.get('name')!=asset_name or not response.get('id'):
        raise RuntimeError('Draft model upload response invalid; manual recovery required')
    # Save receipt first (upload accepted) to avoid blind repeated POST.
    receipt.write_text(json.dumps({'asset':asset_name,'sha256':current_sha,
                        'bytes':len(raw),'asset_id':response['id'],
                        'verified':False},sort_keys=True)+'\n')
    for sec in (0,1,2,3,5,8,13):
        if sec:time.sleep(sec)
        for a in assets(release,token).values():
            if a['name']==asset_name:
                restored=get_private_asset(a,token)
                if restored!=raw:raise RuntimeError('Private uploaded model bytes mismatch')
                receipt.write_text(json.dumps({'asset':asset_name,'sha256':current_sha,
                  'bytes':len(raw),'asset_id':a['id'],'verified':True},sort_keys=True)+'\n')
                return 'PRIVATE_UPLOADED_VERIFIED'
    raise RuntimeError('Private draft model PUT accepted but not byte-verified; halt before test')

def preflight_private_backups(token,release,receipt_dir):
    """Test draft asset upload AND authenticated binary read-back before dataset loading."""
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('G3-backup-preflight.txt',
                   'This probe contains no source data, model, token or traffic identifiers.')
    result=archive_model(buf.getvalue(),token=token,release=release,
                         asset_name='G3_BACKUP_PREFLIGHT_PRIVATE.zip',
                         receipt_dir=receipt_dir)
    return result

def safe_metrics(record):
    allowed={'schema','experiment','protocol_blob_sha','fold','seed','train_days',
        'validation_day','test_day','source_hashes','features_sha256','hyperparameters',
        'training_rows','validation_rows','test_rows','train_attack_rate',
        'validation_selection','b0','b1','b2','model_asset_sha256',
        'private_backup_verified','original_label_provenance','limitations'}
    if set(record)!=allowed or record.get('schema')!='E7_G3_PUBLIC_METRICS_001':
        raise ValueError('Unexpected public metrics schema')
    if record['private_backup_verified'] is not True:
        raise ValueError('Never publish metrics without private model archive verification')
    if not re.fullmatch('[a-f0-9]{64}',record['protocol_blob_sha']):
        raise ValueError('Protocol hash absent')
    if not re.fullmatch('[a-f0-9]{64}',record['model_asset_sha256']):
        raise ValueError('Private model content hash absent')
    assert record['seed'] in (42,43,44,45,46)
    assert record['original_label_provenance']=='UNVERIFIED_ORIGINAL_CICIDS2017_LABELS'
    assert len(record['features_sha256'])==64
    assert len(record['source_hashes'])==8
    assert set(record['b0'])==set(record['b1'])==set(record['b2'])
    for variant in ('b0','b1','b2'):
        metrics=record[variant]
        assert 0<=metrics['fpr']<=1 and 0<=metrics['macro_f1']<=1
        assert metrics['rows']==record['test_rows']
    if any(x in json.dumps(record).lower() for x in (
          '"source_ip"','"destination_ip"','"flow_id"','"github_token"',
          '"raw_predictions"','"private_asset_url"')):
        raise ValueError('Private data accidentally present')
    return (json.dumps(record,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode('utf-8')

def check_content(url,token,data,blob=None,commit=None):
    refs=[]
    if blob:refs.append(ROOT+'/git/blobs/'+blob)
    if commit:refs.append(url+'?ref='+quote(commit,safe=''))
    refs.append(url)
    for link in refs:
        try:j=api('GET',link,token)
        except HTTPError as e:
            if e.code in (404,409,429,500,502,503,504):continue
            raise
        if base64.b64decode(j['content'])!=data:
            raise RuntimeError('Different public metrics already committed: no overwrite')
        return True
    return False

def publish_metrics(record,*,token,output_folder):
    content=safe_metrics(record)
    fid=record['fold'].split('_')[-1]
    name='G3_FOLD_'+fid+'_SEED_'+str(record['seed'])+'.metrics.json'
    folder=Path(output_folder);folder.mkdir(parents=True,exist_ok=True)
    local=folder/name
    if local.exists():
        if local.read_bytes()!=content:
            raise RuntimeError('Conflicting original local metrics; no overwrite')
    else:local.write_bytes(content)
    url=ROOT+'/contents/'+quote(PUBLIC_PREFIX+'/'+name,safe='/')
    if check_content(url,token,content):return 'PUBLIC_VERIFIED_EXISTING'
    try:
        response=api('PUT',url,token,{
            'message':'results(e7-g3): preregistered source-day heldout '+name,
            'branch':'main','content':base64.b64encode(content).decode('ascii')})
    except HTTPError as err:
        if err.code not in (409,422):raise
        for sec in (0,1,2,3,5,8):
            if sec:time.sleep(sec)
            if check_content(url,token,content):return 'PUBLIC_VERIFIED_CONCURRENT'
        raise RuntimeError('Conflicting original report; no overwrite') from err
    blob=response['content']['sha'];commit=response['commit']['sha']
    for sec in (0,1,2,3,5,8,13):
        if sec:time.sleep(sec)
        if check_content(url,token,content,blob,commit):return 'PUBLIC_UPLOADED_VERIFIED'
    raise RuntimeError('Public metrics PUT accepted but not yet verified; no repeat PUT')


def publish_summary(report,*,token,output_folder):
    """Create-only safe fold/seed-level score aggregates, not raw predictions."""
    required={'schema','experiment','protocol_blob_sha','model_count',
              'fold_count','seed_count','fold_summary','limitations'}
    if set(report)!=required or report['schema']!='E7_G3_SEED_SUMMARY_001':
        raise ValueError('Unexpected model score summary schema')
    assert report['model_count']==20 and report['fold_count']==4 and report['seed_count']==5
    assert set(report['fold_summary'])=={'OPENWORLD_DAY_FOLD_'+str(i) for i in range(1,5)}
    assert re.fullmatch('[a-f0-9]{64}',report['protocol_blob_sha'])
    for fold,values in report['fold_summary'].items():
        assert set(values)=={'test_day','n_seeds','b0_macro_f1_mean',
          'b1_macro_f1_mean','b1_macro_f1_std','b2_macro_f1_mean',
          'b2_macro_f1_std','b2_attack_recall_mean','b2_fpr_mean'}
        assert values['n_seeds']==5
    content=(json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
    folder=Path(output_folder);folder.mkdir(parents=True,exist_ok=True)
    filename='G3_FOUR_FOLD_FIVE_SEED_SUMMARY.json'
    path=folder/filename
    if path.exists():
        if path.read_bytes()!=content:raise RuntimeError('Original summary differs')
    else:path.write_bytes(content)
    address=ROOT+'/contents/'+quote(PUBLIC_PREFIX+'/'+filename,safe='/')
    if check_content(address,token,content):return 'SUMMARY_ALREADY_VERIFIED'
    try:
        out=api('PUT',address,token,{'message':'results(e7-g3): frozen four-fold five-seed baseline summary',
          'branch':'main','content':base64.b64encode(content).decode('ascii')})
    except HTTPError as err:
        if err.code not in (409,422):raise
        if check_content(address,token,content):return 'SUMMARY_VERIFIED_CONCURRENT'
        raise
    blob=out['content']['sha'];commit=out['commit']['sha']
    for sec in (0,1,2,3,5,8,13):
        if sec:time.sleep(sec)
        if check_content(address,token,content,blob,commit):return 'SUMMARY_UPLOADED_VERIFIED'
    raise RuntimeError('Summary PUT may have succeeded. Do not resend before verification')
