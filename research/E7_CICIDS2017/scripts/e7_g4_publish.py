"""Create-only G4 aggregate publisher. No model bytes or rowwise predictions public."""
import base64
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote

ROOT="https://api.github.com/repos/itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX="research/E7_CICIDS2017/results/G4_RISK_ABSTENTION"

def validate(report):
    required={'schema','experiment','parent_experiment','g4_protocol_git_blob','g3_protocol_git_blob',
              'fold','seed','train_days','validation_day','test_day','original_source_hashes',
              'original_model_sha256','features_sha256','original_g3_metrics_blob',
              'original_label_provenance','validation_only_selection','m0_g3_b1_macro_f1',
              'm1_alarm_metrics','m2_selective_metrics','scientific_status','limitations'}
    if set(report)!=required or report.get('schema')!='E7_G4_CALIBRATION_REPORT_001':
        raise ValueError('Unsupported G4 public schema')
    assert report['experiment']=='E7-G4-RISK-ABSTENTION-001'
    assert report['parent_experiment']=='E7-G3-OPENWORLD-DAYHOLDOUT-001'
    for key in ('g4_protocol_git_blob','g3_protocol_git_blob','original_g3_metrics_blob'):
        assert re.fullmatch('[a-f0-9]{40}',report[key]),key
    for key in ('original_model_sha256','features_sha256'):
        assert re.fullmatch('[a-f0-9]{64}',report[key]),key
    assert re.fullmatch('OPENWORLD_DAY_FOLD_[1-4]',report['fold'])
    assert report['seed'] in (42,43,44,45,46)
    assert len(report['original_source_hashes'])==8
    assert all(re.fullmatch('[a-f0-9]{64}',v) for v in report['original_source_hashes'].values())
    assert report['original_label_provenance']=='UNVERIFIED_ORIGINAL_CICIDS2017_LABELS'
    assert report['scientific_status']=='RETROSPECTIVE_EXPLORATORY_REUSED_G3_TEST'
    val=report['validation_only_selection']
    assert -0.000001<=val['auto_benign_low']<val['alarm_high']<=1.000001
    assert report['m1_alarm_metrics']['rows']==report['m2_selective_metrics']['rows']
    assert report['limitations'] and len(report['limitations'])>=3
    raw=json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    forbidden=('"raw_predictions"','"flow_id"','"source_ip"','"destination_ip"',
               '"github_token"','"private_model_bytes"','"private_release_url"')
    if any(needle in raw.lower() for needle in forbidden):
        raise ValueError('Private or row-wise data in G4 aggregate')
    return raw.encode()

def checked(api,token,url,data,blob=None,commit=None):
    links=[]
    if blob:links.append(ROOT+'/git/blobs/'+blob)
    if commit:links.append(url+'?ref='+quote(commit,safe=''))
    links.append(url)
    for link in links:
        try:
            item=api('GET',link,token)
        except HTTPError as error:
            if error.code in (404,409,429,500,502,503,504):continue
            raise
        if base64.b64decode(item['content'])!=data:
            raise RuntimeError('Conflicting published G4 artifact; never overwrite')
        return True
    return False

def publish(report,*,api,token,output_dir):
    data=validate(report)
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    fold=report['fold'].rsplit('_',1)[-1]
    name='G4_FOLD_'+fold+'_SEED_'+str(report['seed'])+'.json'
    local=folder/name
    if local.exists():
        if local.read_bytes()!=data:raise RuntimeError('Local G4 immutable artifact changed')
    else:local.write_bytes(data)
    url=ROOT+'/contents/'+quote(PREFIX+'/'+name,safe='/')
    if checked(api,token,url,data):return 'G4_VERIFIED_EXISTING'
    try:
        response=api('PUT',url,token,{'branch':'main',
          'message':'results(e7-g4): exploratory reused-G3-test calibration '+name,
          'content':base64.b64encode(data).decode()})
    except HTTPError as e:
        if e.code not in (409,422):raise
        if checked(api,token,url,data):return 'G4_VERIFIED_CONCURRENT'
        raise RuntimeError('G4 publish conflict; no overwrite') from e
    blob=response['content']['sha'];commit=response['commit']['sha']
    for sec in (0,1,2,3,5,8,13):
        if sec:time.sleep(sec)
        if checked(api,token,url,data,blob,commit):return 'G4_UPLOADED_VERIFIED'
    raise RuntimeError('G4 PUT accepted but not verified; do not retry blindly')

def validate_summary(record):
    required={'schema','experiment','g4_protocol_git_blob','n_reports','n_folds','n_seeds',
              'fold_results','scientific_status','limitations'}
    if set(record)!=required or record['schema']!='E7_G4_SUMMARY_001':
        raise ValueError('G4 summary schema')
    assert re.fullmatch('[a-f0-9]{40}',record['g4_protocol_git_blob'])
    assert record['n_reports']==20 and record['n_folds']==4 and record['n_seeds']==5
    assert set(record['fold_results'])=={'OPENWORLD_DAY_FOLD_'+str(x) for x in range(1,5)}
    assert record['scientific_status']=='RETROSPECTIVE_EXPLORATORY_REUSED_G3_TEST'
    return (json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def publish_summary(record,*,api,token,output_dir):
    data=validate_summary(record)
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    local=folder/'G4_FOUR_FOLD_FIVE_SEED_SUMMARY.json'
    if local.exists():
        if local.read_bytes()!=data:raise RuntimeError('Local summary changed')
    else:local.write_bytes(data)
    url=ROOT+'/contents/'+quote(PREFIX+'/'+local.name,safe='/')
    if checked(api,token,url,data):return 'G4_SUMMARY_VERIFIED_EXISTING'
    response=api('PUT',url,token,{'branch':'main',
       'message':'results(e7-g4): exploratory four-fold calibration summary',
       'content':base64.b64encode(data).decode()})
    for sec in (0,1,2,3,5,8,13):
        if sec:time.sleep(sec)
        if checked(api,token,url,data,response['content']['sha'],response['commit']['sha']):
            return 'G4_SUMMARY_UPLOADED_VERIFIED'
    raise RuntimeError('G4 summary PUT maybe accepted; never repeat blindly')
