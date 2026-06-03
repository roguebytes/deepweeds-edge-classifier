# Hailo AI HAT+ — setup + DeepWeeds inference benchmark runbook

Fit the **Hailo AI HAT+** to the Pi 5, run the DeepWeeds classifier on the NPU, and capture the accelerator number that completes the edge story. The payoff is a **single results table** comparing the *same* model across CPU and NPU:

| Target | Precision | Throughput |
|---|---|---|
| Mac CPU (ONNX Runtime) | fp32 | (have it) |
| Pi 5 CPU (ONNX Runtime) | fp32 | (task #13) |
| Pi 5 CPU (ONNX Runtime) | INT8 dynamic | (task #13 — expected *slower*) |
| **Pi 5 + Hailo NPU** | **INT8 static** | **this runbook** |

> **The narrative this proves:** fp32 on CPU is fine, **dynamic INT8 on CPU is *slower*** (ORT has no fast int8 depthwise kernels), and the INT8 speed-up only pays off on a **dedicated accelerator** — the Hailo. Quantified, on real hardware, at a stated accuracy cost. That's the portfolio point.

**Prerequisite:** the Pi 5 is already set up and the **CPU baseline captured** first — see [`pi5_benchmark.md`](pi5_benchmark.md). Do the CPU numbers *before* fitting the HAT+ so you have the "before."

---

## Phase 1 — Fit the AI HAT+ (hardware)

1. **Power off and unplug** the Pi.
2. The AI HAT+ mounts **above** the board and ships with a **PCIe FFC ribbon cable**, **GPIO stacking header**, and **standoffs** sized to clear the **Active Cooler** — the cooler stays fitted, the HAT sits over it.
3. Fit the standoffs to the Pi. Lift the latch on the Pi's **PCIe connector** (next to the USB-C), slide the **FFC cable** in the correct orientation (contacts facing the right way per the HAT+ guide), press the latch down. Repeat at the HAT+ end.
4. Seat the HAT+ onto the 40-pin header + standoffs, screw down. Re-confirm the **Active Cooler fan cable** is still on its header.
5. Power up.

## Phase 2 — Install the Hailo software (on the Pi)

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install hailo-all          # kernel driver + firmware + HailoRT + Tappas
sudo reboot
```

Verify after reboot:
```bash
hailortcli fw-control identify      # device, firmware, and Device Architecture
lspci | grep -i hailo               # PCIe enumeration
dmesg | grep -i hailo               # driver load
```
- **Confirmed hardware: Hailo-8 (26 TOPS)** — `identify` should report **Device Architecture `HAILO8`**. This sets the compile **`--hw-arch hailo8`** flag in Phase 4. (The 13-TOPS variant would report `HAILO8L`.)
- **PCIe Gen 3 is auto-enabled on the AI HAT+** — no `config.txt`/`raspi-config` change needed (this differs from the older M.2 AI Kit). Sanity-check the link if curious: `sudo lspci -vv | grep -i LnkSta` → expect **8GT/s** (Gen3).
- Keep a note of `hailortcli --version` — the **compiler suite version in Phase 4 must match this major version**, or the `.hef` won't load.

## Phase 3 — Fast sanity number (precompiled Model Zoo HEF) — a Hailo FPS *today*

You don't need to compile anything to prove the board and get a ballpark accelerator number.

1. Download a **precompiled MobileNet `.hef`** from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) — the **`hailo8`** build (your 26-TOPS arch).
2. Benchmark it on-device:
   ```bash
   hailortcli benchmark mobilenet_v3.hef     # auto-generates inputs → FPS + latency
   ```
- **Reality check on FPS:** the Pi 5 exposes **PCIe Gen3 ×1 (single lane)**; Hailo's official Model Zoo benchmarks use multi-lane rigs, so your numbers will be **lower than the headline** — that's expected. **Quote your own measured FPS, never the Model Zoo's.**

This confirms the HAT+ works end-to-end and gives an early "accelerator works" datapoint while you set up the real compile.

## Phase 4 — Compile YOUR DeepWeeds model → `.hef` (the real comparison)

**Host requirement (the catch):** the Hailo **Dataflow Compiler** / `hailomz` runs **only on x86_64 Linux** (≥16 GB RAM, **32 GB recommended** — calibration is memory-heavy). It does **not** run on the Pi (ARM) or natively on your **Apple-Silicon Mac**. Pick a host:
- a **cloud x86 Linux VM** (Ubuntu 22.04, ≥32 GB) for a one-off — simplest if you don't own an x86 box;
- any **x86_64 Ubuntu PC**;
- Windows-x86 via **WSL2 Ubuntu** also works.

On the x86 Linux host:
1. **Hailo Developer Zone** account (free) → download the **Hailo AI Software Suite** (Docker image bundles DFC + Model Zoo + HailoRT). **Match the suite's HailoRT major to the Pi's** (`hailortcli --version` from Phase 2).
2. **Calibration set:** ~**64 representative DeepWeeds images**, a *class-balanced* subset of your **train** split, 256×256, into `calib/`. (Hailo uses these to choose INT8 scales — real, diverse images, not random noise.)
3. **Use the fp32 ONNX** you already export (`model.onnx` from `export.py`) — **not** the `int8` one; Hailo does its **own static INT8 quantization** via calibration.
4. **Compile** (MobileNetV3-Large is a known Model Zoo arch):
   ```bash
   hailomz compile mobilenet_v3_large \
     --ckpt model.onnx \
     --hw-arch hailo8 \           # 26-TOPS Hailo-8 (the 8L would be hailo8l)
     --calib-path calib/ \
     --classes 9
   # → mobilenet_v3_large.hef
   ```
   - **MobileNetV3 quirk (budget iteration here):** its **squeeze-excite + hard-swish** ops occasionally trip the parser. If compile errors on unsupported ops, either (a) use the Model Zoo's `mobilenet_v3` parse config / specify end-node names, or (b) fall back to a **Hailo-friendly backbone** (`resnet18` / `mobilenet_v2`) retrained on DeepWeeds — and **note the swap honestly** in the writeup. This is the only finicky step.
5. **Copy the `.hef` to the Pi:** `scp mobilenet_v3_large.hef <user>@pi5.local:~/`

## Phase 5 — Run + benchmark on the Pi (the payoff)

```bash
hailortcli benchmark mobilenet_v3_large.hef    # Hailo FPS + latency for YOUR model
```

**Also measure accuracy after static quantization** — this is the honest half of the story (INT8 *static* quant can shift accuracy, unlike the fp32 path you measured top-1 on):
- Run the `.hef` over your **held-out test split** (HailoRT Python API in a small infer script, or `hailomz eval` on the host) and report **Top-1 vs the fp32 97.15 %**.

## Phase 6 — Fill the results table (portfolio output)

Add the Hailo row to the README results table and write the one-liner:

| Model / target | Precision | Latency | Throughput | Top-1 |
|---|---|---|---|---|
| MobileNetV3-L · Mac CPU (ORT) | fp32 | 5.83 ms | 171.6 FPS | 97.15 % |
| MobileNetV3-L · Pi 5 CPU (ORT) | fp32 | _task #13_ | _task #13_ | 97.15 % |
| MobileNetV3-L · Pi 5 CPU (ORT) | INT8 dyn | _task #13_ | _task #13_ | — |
| **MobileNetV3-L · Pi 5 + Hailo** | **INT8 static** | _this_ | _this_ | _re-measured_ |

> e.g. "On the Pi 5, fp32 CPU runs at **X FPS**; dynamic INT8 on the same CPU is *slower* (**W FPS**); the **Hailo NPU runs the same network at Y FPS** — a **Y/X×** speed-up at **Z %** top-1 (vs 97.15 % fp32). The INT8 win needs the accelerator, not just a smaller model."

---

## Gotchas
- **Version mismatch** between the compiling suite (host) and the Pi's HailoRT is the #1 "`.hef` won't load" cause — keep majors aligned.
- **PCIe ×1** means you won't hit Model Zoo headline FPS — set expectations, quote measured.
- **Thermals:** the NPU + Pi under sustained load still rely on the **Active Cooler**; the HAT sits above it, airflow is fine, but re-check `vcgencmd measure_temp` during long runs.
- **Accuracy honesty:** report the post-quantization top-1, not the fp32 number, for the deployed Hailo model.

## Why this matters beyond DeepWeeds
This is the **same deployment path** the autonomy demos use onboard: the [`uav-detect-and-track`](../../uav-detect-and-track/) YOLO detector and the Demo 5 maritime detector ([`maritime_vessel_detect_spec.md`](../../maritime_vessel_detect_spec.md)) would each be ONNX → Hailo `.hef` for the **sub-2 kg Lark's** Pi 5 + Hailo payload. Proving the toolchain on the simple classifier first de-risks all of them.

## Reference
- [Raspberry Pi — AI HAT+ docs](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)
- [hailo-ai/hailo-rpi5-examples](https://github.com/hailo-ai/hailo-rpi5-examples) (install + examples)
- [hailo-ai/hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo) (precompiled HEFs + `hailomz`)
