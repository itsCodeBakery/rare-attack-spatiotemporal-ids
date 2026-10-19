"""Create checksum-verified M1 private archive; never publish to GitHub as a public commit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import os


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def freeze(seed_dir,archive_dir):
    seed_dir=Path(seed_dir).resolve();archive_dir=Path(archive_dir).resolve()
    summary=seed_dir/'final_public_summary.json'
    if not summary.is_file():raise ValueError('M1 not complete: refusing partial final archive')
    report=json.loads(summary.read_text())
    if (report.get('official_test_accessed') is not False or
        report.get('status')!='COMPLETED_PENDING_INDEPENDENT_RESULT_AUDIT'):
        raise ValueError('M1 completion metadata violates no-test rules')
    private=seed_dir/'sealed_outer_predictions_PRIVATE.npz'
    if not private.is_file() or sha256(private)!=report['private_predictions_sha256']:
        raise ValueError('M1 frozen original predictions digest mismatch')
    files=sorted(p for p in seed_dir.rglob('*') if p.is_file() and not p.is_symlink())
    if not files or any('testing' in p.name.lower() for p in files):
        raise ValueError('Suspicious M1 archive file')
    archive_dir.mkdir(parents=True,exist_ok=True)
    archive=archive_dir/f'M1-CONTEXT-RISK-001_seed{report["seed"]}_PRIVATE.tar.gz'
    manifest=[{'path':p.relative_to(seed_dir).as_posix(),
               'size':p.stat().st_size,'sha256':sha256(p)} for p in files]
    manifest_file=archive_dir/'M1_PRIVATE_MANIFEST.json'
    manifest_file.write_text(json.dumps({'source_test_accessed':False,
        'files':manifest},indent=2)+'\n')
    if archive.exists():raise ValueError('Immutable archive exists: refuse overwrite')
    partial=archive.with_suffix('.partial')
    try:
        with tarfile.open(partial,'w:gz') as archive_out:
            archive_out.add(manifest_file,arcname='manifest.json',recursive=False)
            for p in files:
                archive_out.add(p,arcname='seed/'+p.relative_to(seed_dir).as_posix(),recursive=False)
        os.replace(partial,archive)
    finally:
        partial.unlink(missing_ok=True)
    checksum=sha256(archive)
    sidecar=archive_dir/(archive.name+'.sha256')
    sidecar.write_text(f'{checksum}  {archive.name}\n')
    print('PRIVATE ARCHIVE',archive,'SHA256',checksum,flush=True)
    return archive,sidecar


def upload_private_draft(archive,checksum,token):
    """Optional: an unpublished GitHub draft Release in the user's existing repo."""
    import requests
    repo='itsCodeBakery/rare-attack-spatiotemporal-ids'
    archive=Path(archive)
    h={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
       'X-GitHub-Api-Version':'2022-11-28'}
    base=f'https://api.github.com/repos/{repo}'
    release_tag=f'm1-context-risk-001-{archive.stem.replace(".","-")}-frozen-private'
    def go(method,url,**opts):
        r=requests.request(method,url,headers=h,timeout=(45,300),**opts)
        r.raise_for_status()
        return r.json()
    releases=go('GET',base+'/releases',params={'per_page':100})
    release=next((x for x in releases if x['tag_name']==release_tag),None)
    if release is None:
        release=go('POST',base+'/releases',json={'tag_name':release_tag,
                    'target_commitish':'main','draft':True,'prerelease':True,
                    'name':f'M1 PRIVATE ORIGINAL seed artifact {archive.name}',
                    'body':'Do not publish; internal train-derived model/probabilities. '+checksum})
    if not release['draft']:raise ValueError('Refuse uploading M1 to public Release')
    assets={x['name']:x for x in release.get('assets',[])}
    for p in (archive,archive.with_suffix(archive.suffix+'.sha256')):
        s=sha256(p);prior=assets.get(p.name)
        if prior:
            if prior.get('size')!=p.stat().st_size or prior.get('digest')!='sha256:'+s:
                raise ValueError('Remote existing private artifact mismatch')
        else:
            upload_url=release['upload_url'].split('{')[0]
            with p.open('rb') as f:
                r=requests.post(upload_url,headers=dict(h,**{
                    'Content-Type':'application/octet-stream'}),
                    params={'name':p.name},data=f,timeout=(45,3600))
            r.raise_for_status()
            created=r.json()
            if created.get('size')!=p.stat().st_size or created.get('digest') not in (None,'sha256:'+s):
                raise ValueError('Upload size/hash mismatch')
    print('VERIFIED UNPUBLISHED GITHUB DRAFT',release['html_url'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed-dir',required=True)
    parser.add_argument('--archive-dir',required=True)
    parser.add_argument('--draft-upload',action='store_true')
    args=parser.parse_args()
    archive,side=freeze(args.seed_dir,args.archive_dir)
    if args.draft_upload:
        token=os.environ.get('GITHUB_TOKEN')
        if not token:raise ValueError('No GITHUB_TOKEN: local archive created, draft not uploaded')
        upload_private_draft(archive,sha256(archive),token)
