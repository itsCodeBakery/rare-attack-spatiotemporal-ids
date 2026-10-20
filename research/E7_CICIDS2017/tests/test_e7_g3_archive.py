"""No-network tests: G3 models remain draft-only and uploads are immutable."""
import base64,hashlib,importlib.util,io,json,zipfile
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]/'scripts/e7_g3_archive.py'
spec=importlib.util.spec_from_file_location('g3_pub',ROOT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def draft():
    return {'draft':True,'tag_name':mod.TAG,'id':543,
            'upload_url':'https://uploads.github.com/repos/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/543/assets{?name,label}'}
def zipbytes():
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('test.txt','Never includes network traffic')
    return out.getvalue()
def raw_report():
    metric={'threshold':0.5,'macro_f1':0.6,'fpr':0.1,'rows':3,
            'tn':1,'tp':1,'fp':1,'fn':0,'auroc':0.8,'average_precision':0.75}
    return {'schema':'E7_G3_PUBLIC_METRICS_001','experiment':'E7-G3-OPENWORLD-DAYHOLDOUT-001',
            'protocol_blob_sha':'a'*64,'fold':'OPENWORLD_DAY_FOLD_1','seed':42,
            'train_days':['Monday','Thursday','Friday'],
            'validation_day':'Wednesday','test_day':'Tuesday',
            'source_hashes':{'f'+str(i):'f'*64 for i in range(8)},
            'features_sha256':'b'*64,'hyperparameters':{},
            'training_rows':10,'validation_rows':5,'test_rows':3,
            'train_attack_rate':0.2,'validation_selection':{},
            'b0':dict(metric),'b1':dict(metric),'b2':dict(metric),
            'model_asset_sha256':'c'*64,'private_backup_verified':True,
            'original_label_provenance':'UNVERIFIED_ORIGINAL_CICIDS2017_LABELS',
            'limitations':['Original labels have not been independently corrected']}

def test_public_aggregate_metrics_safe_and_private_backup_required():
    d=raw_report()
    assert mod.safe_metrics(d)
    d['private_backup_verified']=False
    with pytest.raises(ValueError,match='private model'):mod.safe_metrics(d)
    d=raw_report();d['source_ip']='10.0.0.2'
    with pytest.raises(ValueError):mod.safe_metrics(d)
    d=raw_report();d['model_asset_sha256']='bad'
    with pytest.raises(ValueError):mod.safe_metrics(d)

def test_draft_must_stay_unpublished(monkeypatch):
    def fake(method,url,token,payload=None):
        if method=='GET' and '/releases/tags/' in url:
            return {**draft(),'draft':False}
        raise AssertionError('unexpected action')
    monkeypatch.setattr(mod,'api',fake)
    with pytest.raises(RuntimeError,match='NOT a private draft'):
        mod.draft_release('FAKE')

def test_unpublished_release_creation_and_preflight(monkeypatch):
    calls=[]
    def fake(method,url,token,payload=None):
        calls.append(method)
        if method=='GET' and '/releases/tags/' in url:
            from urllib.error import HTTPError
            raise HTTPError(url,404,'Not Found',None,None)
        if method=='GET' and url.endswith('/releases?per_page=100'):return []
        if method=='POST':assert payload['draft'] is True;return draft()
        if method=='GET' and '/assets?' in url:return []
        raise AssertionError(method)
    monkeypatch.setattr(mod,'api',fake)
    assert mod.draft_release('FAKE')['draft'] is True
    assert calls==['GET','GET','POST','GET']

def test_backup_probe_preflight_saved_and_verified(monkeypatch,tmp_path):
    uploaded={}
    model=zipbytes()
    def fake_api(method,url,token,payload=None):
        if method=='GET' and '/assets?' in url:
            if not uploaded:return []
            return [{'name':'G3_BACKUP_PREFLIGHT_PRIVATE.zip','id':9,'size':len(uploaded['bytes']),
                     'digest':'sha256:'+mod.sha(uploaded['bytes'])}]
        raise AssertionError(method)
    def fake_http(method,url,token,data=None,headers=None,parse_json=True):
        assert method=='POST'
        assert 'uploads.github.com' in url
        assert data.startswith(b'PK\x03\x04')
        uploaded['bytes']=data
        return {'name':'G3_BACKUP_PREFLIGHT_PRIVATE.zip','id':9}
    monkeypatch.setattr(mod,'api',fake_api)
    monkeypatch.setattr(mod,'http',fake_http)
    monkeypatch.setattr(mod,'get_private_asset',lambda asset,token:uploaded['bytes'])
    status=mod.preflight_private_backups('FAKE',draft(),tmp_path)
    assert status=='PRIVATE_UPLOADED_VERIFIED'
    assert mod.preflight_private_backups('FAKE',draft(),tmp_path)=='PRIVATE_EXISTING_VERIFIED'

def test_remote_model_asset_mismatch_refuses_continue(monkeypatch,tmp_path):
    model=zipbytes()
    monkeypatch.setattr(mod,'assets',lambda release,token:{
        'G3_FOLD_1_SEED_42_PRIVATE.zip':{'id':10,'size':5}})
    monkeypatch.setattr(mod,'get_private_asset',lambda asset,token:b'WRONG')
    with pytest.raises(RuntimeError,match='Conflicting private model'):
        mod.archive_model(model,token='FAKE',release=draft(),
                          asset_name='G3_FOLD_1_SEED_42_PRIVATE.zip',receipt_dir=tmp_path)

def test_existing_public_results_cannot_be_replaced(monkeypatch,tmp_path):
    monkeypatch.setattr(mod,'api',lambda method,url,token,payload=None:{
        'content':base64.b64encode(b'different metric file').decode('ascii')})
    with pytest.raises(RuntimeError,match='Different public metrics'):
        mod.publish_metrics(raw_report(),token='FAKE',output_folder=tmp_path)
