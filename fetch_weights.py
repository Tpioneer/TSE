"""Fetch verified official files into assets/spk_emb_causal_100; skip valid local files."""
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent
URL='https://drive.usercontent.google.com/download?id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC&export=download&confirm=t'
HASHES={'avg_model.pt':'42b73219eefefcbba4b5da0dcb89dce292359c9f93b1a951c7b6d196da2155b3',
        'config.yaml':'80820eb78de6639160f7124d09cbacbc8397e321d8075721e89edb7cec50a0fa'}


def valid(path, digest):
    if not path.is_file():
        return False
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()==digest


def main():
    directory=ROOT/'assets/spk_emb_causal_100'
    directory.mkdir(parents=True,exist_ok=True)
    if all(valid(directory/name,h) for name,h in HASHES.items()):
        print("Verified official checkpoint and configuration.")
        return
    archive=directory.parent/'official-download'
    if not zipfile.is_zipfile(archive):
        tmp=archive.with_suffix('.part')
        urllib.request.urlretrieve(URL,tmp)
        if not zipfile.is_zipfile(tmp):
            raise ValueError("Expected official ZIP; check Google Drive download access.")
        tmp.replace(archive)
    with zipfile.ZipFile(archive) as src:
        for name,digest in HASHES.items():
            matches=[m for m in src.namelist() if tuple(Path(m).parts[-2:])==('spk_emb_causal_100',name)]
            if len(matches)!=1:
                raise ValueError(f"Missing or ambiguous archive entry for {name}")
            tmp=directory/(name+'.part')
            with src.open(matches[0]) as fin, tmp.open('wb') as fout:
                shutil.copyfileobj(fin,fout)
            if not valid(tmp,digest):
                raise ValueError(f"Hash mismatch: {name}")
            tmp.replace(directory/name)


if __name__=='__main__':
    main()
