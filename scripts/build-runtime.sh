#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
revision=8f5927397d1b7c87e6724cb1ddbde956a7f34cab
mkdir -p vendor
if [ ! -d vendor/fastrpc/.git ]; then git clone https://github.com/qualcomm/fastrpc.git vendor/fastrpc; fi
git -C vendor/fastrpc checkout --detach "$revision"
[ -z "$(git -C vendor/fastrpc status --porcelain --untracked-files=no)" ] || { echo 'FastRPC source has local changes.' >&2; exit 1; }
(cd vendor/fastrpc && GITCOMPILE_NO_MAKE=1 CFLAGS='-O2 -Wall -pipe' ./gitcompile --prefix=/opt/unoq-dsp --with-config-base-dir=/opt/unoq-dsp/config && make -j2)
./scripts/build-kernels.sh
cd runtime
gcc -O3 -Wall -Wextra -shared -fPIC -I ../vendor/fastrpc/inc bridge.c -L ../vendor/fastrpc/src/.libs '-Wl,-rpath,$ORIGIN' -ladsprpc -lpthread -o ../prebuilt/aarch64/libuno_backend.so
cd ..
cp vendor/fastrpc/src/adsprpcd prebuilt/aarch64/
cp vendor/fastrpc/src/.libs/libadsprpc.so.1.0.0 prebuilt/aarch64/libadsprpc.so.1
cp vendor/fastrpc/src/.libs/libadsp_default_listener.so.1.0.0 prebuilt/aarch64/libadsp_default_listener.so
patchelf --set-rpath '$ORIGIN' prebuilt/aarch64/libadsp_default_listener.so
strip prebuilt/aarch64/*
python3 scripts/check_release.py --write
