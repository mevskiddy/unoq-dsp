#!/usr/bin/env python3
"""Fetch the matching upstream user-PD shell; does not replace DSP firmware."""
import hashlib
from pathlib import Path
import urllib.request
root = Path(__file__).resolve().parents[1]
url = ('https://raw.githubusercontent.com/linux-msm/hexagon-dsp-binaries/'
       '53eafda6d174ad7a7a965215cd0b224e855620de/'
       'qcm2290/Thundercomm/RB1/ADSP.VT.5.4.1.c4-00013-KAMORTA-1/fastrpc_shell_0')
expected = '2c3ab4cd6a65ec797351a235f62e58342e84b1e70865c6f65983c32769536c9b'
data = urllib.request.urlopen(url, timeout=60).read()
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit('DSP shell checksum mismatch')
path = root/'.state/fastrpc_shell_0'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_bytes(data)
print(path)
