#!/usr/bin/env python3
"""Verify bundled runtime/model artifacts, or record a new developer build."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--write', action='store_true')
a = p.parse_args()
required = {
 'prebuilt/aarch64/adsprpcd': 183,
 'prebuilt/aarch64/libuno_backend.so': 183,
 'prebuilt/aarch64/libadsprpc.so.1': 183,
 'prebuilt/aarch64/libadsp_default_listener.so': 183,
 'prebuilt/hexagon-v66/libuno_graph.so': 164,
 'models/yolo26n.pt': None,
 'models/mobilenetv4-small.uqmodel': None,
 'models/yolo26n-320.uqmodel': None,
}
actual = {}
for name, machine in required.items():
 data = (root/name).read_bytes()
 if machine is not None:
  if data[:4] != b'\x7fELF' or struct.unpack_from('<H',data,18)[0] != machine:
   raise SystemExit(f'Incorrect executable architecture: {name}')
 actual[name] = hashlib.sha256(data).hexdigest()
manifest = root/'SHA256SUMS.json'
if a.write:
 manifest.write_text(json.dumps(actual,indent=2)+'\n')
else:
 expected = json.loads(manifest.read_text())
 if actual != expected: raise SystemExit('Release checksum mismatch; regenerate after an intentional rebuild with --write.')
print(f'Verified {len(actual)} runtime and model artifacts.')
