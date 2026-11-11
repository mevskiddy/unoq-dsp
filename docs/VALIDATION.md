# Release verification

Verified on Arduino UNO Q, Debian 13 ARM64, kernel `6.16.0-geffa8626771a`.

- Installed into a fresh Python environment and `/opt/unoq-dsp`.
- Ran the bundled classifier on a cat photo and the detector on a street photo, including annotated-image output.
- Completed 30 warm blocking-RPC inferences per model after five warm-ups.
- Verified automatic listener shutdown after each command.
- Rebuilt the Hexagon module from the shipped kernel source; its SHA-256 matches the shipped binary exactly.
- Passed 26 package, preprocessing, decoding and reference-kernel tests.
- Validated device-tree overlay generation against the stock UNO Q DTB without changing the active boot entry during packaging.

| Model | Median invocation | Client-process CPU time |
|---|---:|---:|
| MobileNetV4-Small | 50.05 ms | 1.14 ms |
| YOLO26n | 192.62 ms | 3.19 ms |

Invocation timings include transport and output handling, exclude image preprocessing, prediction decoding and model load. CPU time refers to the calling process.

`SHA256SUMS.json` records the shipped binaries, model bundles and original detector checkpoint. Run `python3 scripts/check_release.py` to verify them.
