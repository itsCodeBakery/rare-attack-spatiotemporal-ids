"""Source-byte encoding gate for external CICIDS2017 CSVs (no reading private flow rows).

Never changes source bytes, never uses errors='replace'/'ignore'. Strict UTF-8
is preferred, otherwise strict cp1252 is accepted only with a corresponding
explicit review flag; anything else fails closed. The same original file SHA256
is used for all source audits and their public artifacts.
"""
from __future__ import annotations
import codecs
import hashlib
from pathlib import Path

def source_encoding_gate(path, block_size=1024*1024):
    path=Path(path)
    if not path.is_file() or path.suffix.lower()!='.csv':
        raise ValueError("Expected an existing external CSV")
    if block_size < 256:
        raise ValueError('Invalid block size')
    size=path.stat().st_size
    if not size:raise ValueError('Empty source file')

    def check(codec):
        decoder=codecs.getincrementaldecoder(codec)('strict')
        try:
            with path.open('rb') as fd:
                for block in iter(lambda:fd.read(block_size),b''):
                    decoder.decode(block,final=False)
            decoder.decode(b'',final=True)
        except UnicodeError:
            return False
        return True

    with path.open('rb') as f: bom=f.read(3)==b'\xef\xbb\xbf'
    utf8=check('utf-8-sig')
    if utf8:
        return {'source_encoding':'utf-8-sig','encoding_status':'STRICT_UTF8_VERIFIED',
                'requires_manual_review':False,'byte_count':size,
                'original_csv_sha256':file_sha(path)}
    # A mixed UTF-8 BOM + non UTF8 payload requires manual review. Do not
    # silently treat it as cp1252 or invent damaged utf8 fragments.
    if bom:raise ValueError('BOM_MIXED_ENCODING: UTF8 BOM plus invalid UTF8; manual source inspection required')
    if check('cp1252'):
        return {'source_encoding':'cp1252',
                'encoding_status':'UTF8_FAILED_STRICT_CP1252_FULL_FILE_VALIDATED',
                'requires_manual_review':True,'byte_count':size,
                'original_csv_sha256':file_sha(path)}
    raise ValueError('ENCODING_UNRESOLVED: neither strict UTF-8 nor Windows-1252 decoded all source bytes. No loss-tolerant replacement allowed.')

def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()
