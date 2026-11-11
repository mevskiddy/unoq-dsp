#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
[ "$(uname -m)" = aarch64 ] || { echo 'Run this installer on the UNO Q.' >&2; exit 1; }
cd "$root"
python3 scripts/check_release.py
sudo apt-get update
sudo apt-get install -y python3-venv libyaml-0-2 libbsd0 libmd0
python3 -m venv .venv
.venv/bin/python -m pip install -c requirements-runtime.txt '.[vision]'
sudo install -d /opt/unoq-dsp/bin /opt/unoq-dsp/lib /opt/unoq-dsp/dsp /opt/unoq-dsp/config/conf.d
sudo install -m 755 prebuilt/aarch64/adsprpcd /opt/unoq-dsp/bin/
for f in prebuilt/aarch64/*.so*; do sudo install -m 755 "$f" /opt/unoq-dsp/lib/; done
sudo install -m 644 prebuilt/hexagon-v66/libuno_graph.so /opt/unoq-dsp/dsp/
sudo install -m 644 provision/unoq.yaml /opt/unoq-dsp/config/conf.d/
printf 'Installed. Run ./run python or see README.md for image inference.\n'
