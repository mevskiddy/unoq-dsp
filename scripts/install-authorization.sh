#!/bin/sh
set -eu
[ "$#" -eq 1 ] || { echo 'Usage: sudo scripts/install-authorization.sh /path/to/testsig-0x....so' >&2; exit 1; }
[ "$(id -u)" -eq 0 ] || { echo 'Run with sudo.' >&2; exit 1; }
serial=$(cat /sys/devices/soc0/serial_number)
expected=$(printf 'testsig-0x%x.so' "$serial")
[ "$(basename -- "$1")" = "$expected" ] || { echo "This board requires $expected" >&2; exit 1; }
[ -s "$1" ] || { echo 'Signature file is empty or missing.' >&2; exit 1; }
install -d /opt/unoq-dsp/dsp
install -m 644 "$1" "/opt/unoq-dsp/dsp/$expected"
printf 'Installed development authorization for this board.\n'
