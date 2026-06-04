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

**First, confirm the HAT is on the PCIe bus** — catches a misseated/backwards FFC ribbon *before* you install anything (`lspci` needs no driver):
```bash
lspci | grep -i hailo
# want: 0001:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor (rev 01)
```
Nothing shown? Power down, re-seat the FFC at **both** ends (orientation is the #1 gotcha), retry.

**Then install the stack and reboot:**
```bash
sudo apt update && sudo apt install -y hailo-all   # kernel driver + firmware + HailoRT + Tappas
sudo reboot
```

**Verify after reboot:**
```bash
hailortcli fw-control identify
```
Confirmed-good output on this rig (2026-06-05):
```
Firmware Version: 4.23.0 (release,app,extended context switch buffer)
Board Name: Hailo-8
Device Architecture: HAILO8
```
- **Hailo-8 (26 TOPS)** → `Device Architecture: HAILO8` → sets the compile **`--hw-arch hailo8`** flag in Phase 4. (The 13-TOPS variant reports `HAILO8L`.)
- **Note the HailoRT/firmware version (here 4.23.0 → HailoRT 4.x).** The Phase 4 compiler suite **and** any precompiled HEF you run **must be HailoRT 4.x-compatible**, or the `.hef` won't load (`hailortcli --version` shows the runtime).
- **PCIe Gen 3 auto-enables on the AI HAT+** — no `config.txt`/`raspi-config` change (differs from the older M.2 AI Kit). Check the link if curious: `sudo lspci -vv | grep -i LnkSta` → expect **8GT/s** (Gen3).

## Phase 3 — Fast sanity number (no compiling) — a Hailo FPS *today*

You don't need to compile anything to prove the board and get a ballpark — **the `hailo-all` install ships demo HEFs already**, no download needed:
```bash
find / -name '*.hef' 2>/dev/null      # typically under /usr/share/hailo-models/
```

**Match the arch suffix to your chip** — a mismatched HEF just errors ("compiled for X, device is Y"):
- **`_h8`** = Hailo-**8** (26 TOPS) ← run these
- `_h8l` = Hailo-8**L** (13 TOPS) · `_h10` = Hailo-**10** ← won't load on a Hailo-8

Benchmark one (auto-generates inputs → FPS + latency):
```bash
hailortcli benchmark /usr/share/hailo-models/yolov6n_h8.hef
```

**Measured on this rig (Pi 5 + Hailo-8, 2026-06-05):** `yolov6n_h8` → **586 FPS** (hw-only *and* streaming) · **3.18 ms** HW latency. No power telemetry (the Pi 5 + AI HAT+ doesn't expose it over PCIe).
- **streaming ≈ hw-only** → the **PCIe Gen3 ×1** link isn't bottlenecking this small model (bigger/higher-res models may show a gap).
- ~10–20× real-time for a nano detector — confirms big headroom for the YOLO-class autonomy/maritime demos.
- ⚠️ **YOLOv6n, not DeepWeeds** — an "NPU validated + detector headroom" datapoint, **not** the classifier comparison. The apples-to-apples DeepWeeds figure needs *your* MobileNetV3 from Phase 4.

*(Alternative: download a precompiled HEF from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) — pick the `hailo8` + HailoRT-4.x build. The preinstalled set is the faster path.)*

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
