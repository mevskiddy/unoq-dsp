# Enable the UNO Q DSP

This setup targets the UNO Q Debian 13 ARM64 image with the `6.16.0-geffa8626771a` kernel and `ADSP.VT.5.4.1.c4-00013-KAMORTA-1` firmware. Run board commands in an SSH terminal from the repository directory.

## 1. Expose FastRPC

```sh
sudo apt-get update
sudo apt-get install -y device-tree-compiler python3-venv curl unzip
sudo python3 scripts/enable-dsp.py
sudo reboot
```

Reconnect and verify:

```sh
ls -l /dev/fastrpc-adsp
```

The script adds the ADSP FastRPC node and context banks 3–7 to a separate device tree, then selects a new boot entry. It retains the stock kernel, firmware and original boot entry. Boards already exposing `/dev/fastrpc-adsp` skip this change.

## 2. Authorize your board

Read its Qualcomm SoC serial number:

```sh
cat /sys/devices/soc0/serial_number
```

On a Linux workstation, obtain [Hexagon SDK 6.4.0.2](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK), which provides the development TestSig generator:

```sh
curl -fL -o Hexagon_SDK_lnx.zip \
  https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.4.0.2/Hexagon_SDK_lnx.zip
unzip -q Hexagon_SDK_lnx.zip
```

Set the path to the extracted SDK and substitute your board’s serial number:

```sh
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.2
python3 "$HEXAGON_SDK_ROOT/tools/elfsigner/elfsigner.py" \
  -t YOUR_BOARD_SERIAL -o ./board-signature
```

Copy the generated `testsig-0x….so` to your UNO Q:

```sh
scp board-signature/testsig-*.so arduino@YOUR_BOARD:~/
```

Back on the board, install the matching signature and fetch the pinned user-process shell:

```sh
sudo scripts/install-authorization.sh ~/testsig-*.so
python3 scripts/fetch-shell.py
sudo install -m 644 .state/fastrpc_shell_0 /opt/unoq-dsp/dsp/
```

The TestSig is generated for each board; it stays outside the repository. The shell download is pinned and checksum-verified. This authorizes a DSP user process alongside the existing firmware.

## 3. Install and run

```sh
./scripts/install.sh
./run python examples/classify.py photo.jpg
./run python examples/detect.py photo.jpg annotated.jpg
```

The runtime is installed under `/opt/unoq-dsp`; Python dependencies are isolated in `.venv`. Opening a session loads `libuno_graph.so` and the selected model onto the DSP through FastRPC. There is no additional firmware upload step, and setup survives reboot.

For an interactive Python prompt:

```sh
./run python
```

## Restore the stock boot entry

```sh
sudo bootctl set-default "$(sudo cat /var/lib/unoq-dsp/boot-backup/stock-entry.txt)"
sudo reboot
```

Boot backups are kept in `/var/lib/unoq-dsp/boot-backup`. The project’s launcher runs the listener only for the lifetime of your application.

## Upstream components

- [Qualcomm FastRPC](https://github.com/qualcomm/fastrpc/tree/8f5927397d1b7c87e6724cb1ddbde956a7f34cab)
- [Matching QCM2290 DSP binaries](https://github.com/linux-msm/hexagon-dsp-binaries/tree/53eafda6d174ad7a7a965215cd0b224e855620de/qcm2290/Thundercomm/RB1/ADSP.VT.5.4.1.c4-00013-KAMORTA-1)
- [Arduino Linux kernel](https://github.com/arduino/linux-qcom)
