from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
for name,expected in json.loads((p/'SHA256SUMS.json').read_text()).items():
 actual=hashlib.sha256((p/name).read_bytes()).hexdigest()
 if actual!=expected:raise RuntimeError('Checksum mismatch: '+name)
print('All packaged files verified.')
