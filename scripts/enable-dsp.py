#!/usr/bin/env python3
"""Create a separate UNO Q FastRPC boot entry, preserving the stock entry."""
import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepare-only', action='store_true', help='Build and validate files without changing boot configuration')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
if Path('/sys/firmware/devicetree/base/model').read_bytes().rstrip(b'\0') != b'Arduino SA,Imola':
    raise SystemExit('This setup targets Arduino UNO Q (Arduino SA,Imola).')
if Path('/dev/fastrpc-adsp').exists() and not args.prepare_only:
    print('FastRPC is already enabled. Continue with DSP authorization.')
    raise SystemExit(0)
if not args.prepare_only and os.geteuid() != 0:
    raise SystemExit('Run with sudo.')
esp = Path('/boot/efi')
base = esp / 'dtb/qcom/qrb2210-arduino-imola.dtb'
entries = []
for entry in (esp/'loader/entries').glob('*.conf'):
    text = entry.read_text()
    if re.search(r'^version\s+' + re.escape(platform.release()) + r'\s*$', text, re.M) and re.search(r'^sort-key\s+debian\s*$', text, re.M):
        entries.append(entry)
if len(entries) != 1:
    raise SystemExit(f'Expected one stock boot entry for {platform.release()}, found {len(entries)}.')
entry = entries[0]
state = root / '.state/boot'
state.mkdir(parents=True, exist_ok=True)
overlay, result = state/'fastrpc.dtbo', state/'unoq-dsp.dtb'
subprocess.run(['dtc','-@','-I','dts','-O','dtb','-o',str(overlay),str(root/'provision/fastrpc.dtso')],check=True)
subprocess.run(['fdtoverlay','-i',str(base),'-o',str(result),str(overlay)],check=True)
node = '/soc@0/remoteproc@ab00000/glink-edge/fastrpc'
compatible = subprocess.check_output(['fdtget',str(result),node,'compatible'],text=True).strip()
assert compatible == 'qcom,fastrpc'
for cb in range(3,8):
    actual = subprocess.check_output(['fdtget','-t','x',str(result),f'{node}/compute-cb@{cb}','reg'],text=True).strip()
    assert int(actual,16) == cb
text = entry.read_text()
text = re.sub(r'^title\s+.*$', 'title      UNO Q DSP', text, flags=re.M)
text = re.sub(r'^sort-key\s+.*$', 'sort-key   unoq-dsp', text, flags=re.M)
text = re.sub(r'^devicetree\s+.*\n?', '', text, flags=re.M)
text = text.rstrip() + '\ndevicetree /dtb/qcom/unoq-dsp.dtb\n'
(state/'unoq-dsp.conf').write_text(text)
if args.prepare_only:
    print(f'Validated overlay and boot entry in {state}; boot configuration unchanged.')
    raise SystemExit(0)
backup = Path('/var/lib/unoq-dsp/boot-backup')
backup.mkdir(parents=True,exist_ok=True)
loader = esp/'loader/loader.conf'
if not (backup/'stock-entry.txt').exists():
    (backup/'stock-entry.txt').write_text(entry.name+'\n')
    if loader.exists(): shutil.copy2(loader,backup/'loader.conf')
    shutil.copy2(entry,backup/entry.name)
shutil.copy2(result,esp/'dtb/qcom/unoq-dsp.dtb')
shutil.copy2(state/'unoq-dsp.conf',esp/'loader/entries/unoq-dsp.conf')
subprocess.run(['bootctl','set-default','unoq-dsp.conf'],check=True)
print('DSP boot entry installed. Reboot, then continue with DSP authorization.')
