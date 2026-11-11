#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
mkdir -p build prebuilt/aarch64 prebuilt/hexagon-v66
cd runtime
. ./vision_flags.sh
for name in graph gemm qnn graph_dsp; do
 clang-19 --target=hexagon -mcpu=hexagonv66 -O3 -ffp-contract=off -fPIC -ffreestanding -fno-stack-protector -fvisibility=hidden -nostdlib $VISION_FLAGS -c "$name.c" -o "../build/$name.o"
done
for name in gemm2x4 requant2; do
 clang-19 --target=hexagon -mcpu=hexagonv66 -fPIC -c "$name.S" -o "../build/$name.o"
done
ld.lld-19 -shared --hash-style=sysv -Bsymbolic -z max-page-size=4096 -z common-page-size=4096 -z norelro -T ../dsp.ld -soname libuno_graph.so ../build/graph.o ../build/gemm.o ../build/gemm2x4.o ../build/requant2.o ../build/qnn.o ../build/graph_dsp.o -o ../prebuilt/hexagon-v66/libuno_graph.so
cd ..
python3 scripts/check_release.py --write
