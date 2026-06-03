# Raspberry Pi 5 — CPU inference benchmark runbook

Re-run the repo's `benchmark.py` on a **Raspberry Pi 5** to get the on-device headline latency/throughput for the DeepWeeds classifier, for **both** the fp32 and INT8-dynamic ONNX models. This is the **CPU baseline** — the "before" number against which the Hailo AI HAT+ accelerator is later compared.

> **What this answers.** On Apple-Silicon CPU we already saw INT8 *dynamic* quantization run **slower** than fp32 (ORT's CPU provider has no fast int8 kernels for MobileNet's depthwise convs). The Pi 5's Cortex-A76 is the same class of CPU, so the expected result here is the same — and that's the point: it shows the INT8 speed-up needs an **accelerator** (Hailo), not just a smaller model. Confirm it with real Pi numbers rather than asserting it.

**No PyTorch on the Pi.** `benchmark.py` only needs `onnxruntime` + `numpy`. Training/export happens on the Mac; the Pi only runs inference. Keep the Pi install tiny.

---

## Phase 0 — One-time Pi setup (~30 min, mostly OS image)

Tickable checklist. **Have in hand:** Pi 5 (8 GB) · official 27 W USB-C PSU · Active Cooler · A1/A2 microSD ≥32 GB · Mac with SD reader. **Don't fit the Hailo HAT+ yet** — that's after the CPU baseline, so the "before" number is clean.

**1 — Fit the Active Cooler** (Pi unplugged)
- [ ] Align the two spring push-pins with the holes either side of the SoC; press each straight down until it clicks
- [ ] Plug the fan lead into the **4-pin JST fan header** (next to USB-C)
- *Why:* sustained inference pins all 4 cores; without active cooling the A76 throttles within seconds and latency drifts mid-run.

**2 — Flash the card** (on the Mac)
- [ ] Install Imager: `brew install --cask raspberry-pi-imager`
- [ ] Imager → **Device:** Raspberry Pi 5 · **OS:** *Raspberry Pi OS (64-bit)* · **Storage:** the card
  - **Must be 64-bit/arm64** — `onnxruntime` has no 32-bit wheels. (Lite is fine for headless.)
- [ ] **Edit Settings** before writing: hostname `pi5` · username+password · Wi-Fi SSID/password + country **AU** · locale **Australia** · **Services → Enable SSH → public-key** (paste `~/.ssh/id_ed25519.pub`; `ssh-keygen -t ed25519` if needed)
- [ ] **Write** → verify → eject

**3 — First boot**
- [ ] Card into Pi → connect the **official 27 W PSU** (no power button — boots on power). *Undervoltage from a weaker supply silently throttles the CPU and ruins the numbers.*
- [ ] Wait ~60–90 s (first boot expands the filesystem + joins Wi-Fi)

**4 — Connect** (from the Mac)
- [ ] `ssh <user>@pi5.local` (accept the host key); if `.local` won't resolve, find the IP on your router

**5 — Update firmware + OS**
- [ ] `sudo apt update && sudo apt full-upgrade -y`
- [ ] `sudo rpi-eeprom-update -a` *(latest bootloader/firmware)*
- [ ] `sudo reboot`

**6 — Health check** (SSH back in)
- [ ] `uname -m` → **`aarch64`** (confirms 64-bit)
- [ ] `vcgencmd get_throttled` → **`throttled=0x0`** (no undervoltage/throttle)
- [ ] `vcgencmd measure_temp` → idle ~40–50 °C · `free -h` → ~8 GB · `python3 --version` → 3.11.x
- *If `get_throttled` ≠ `0x0`: stop and fix it (PSU/cable = undervoltage, cooler seating = thermal) before trusting any benchmark.*

---

## Phase 1 — Project + minimal deps (~5 min)

```bash
# clone (or scp) just what you need
git clone <repo-url> deepweeds-edge-classifier
cd deepweeds-edge-classifier

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy          # that's all benchmark.py imports
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
# expect a version + ['CPUExecutionProvider']  (CPU-only on the Pi — correct for the baseline)
```

## Phase 2 — Put the Pi in a fair, reproducible state

Benchmarks are only comparable if the CPU isn't throttling and isn't idling down. Do this **before every run** and record the results in your notes so the Hailo comparison later is apples-to-apples.

```bash
# 1) Pin the CPU governor to performance (stops on-demand frequency scaling skewing latency)
sudo apt install -y cpufrequtils
sudo cpufreq-set -g performance
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # expect: performance

# 2) Check thermals + throttling
vcgencmd measure_temp        # keep well under 80°C with the Active Cooler
vcgencmd get_throttled       # MUST read throttled=0x0
```
`get_throttled` decoded: `0x0` = clean. Any non-zero means undervoltage (bits 0/16 → fix the PSU/cable) or thermal throttling (bits 2/3 → cooling). **If it's not `0x0`, fix it before trusting any number.**

## Phase 3 — Get the two ONNX models onto the Pi

Export on the **Mac** (needs the trained checkpoint + torch), then copy both files over:

```bash
# on the Mac, in the repo:
python export.py --checkpoint runs/mnv3/best_model.pt --arch mobilenet_v3_large \
    --output runs/mnv3/model.onnx --quantize
# produces:  runs/mnv3/model.onnx          (fp32)
#            runs/mnv3/model.int8.onnx     (INT8 dynamic)

# copy to the Pi (adjust host):
scp runs/mnv3/model.onnx runs/mnv3/model.int8.onnx pi@raspberrypi.local:~/deepweeds-edge-classifier/runs/mnv3/
```

## Phase 4 — Run the benchmark (the actual measurement)

`benchmark.py` already reports mean/std/p50/p95 latency + throughput on `CPUExecutionProvider`. The Pi 5 has **4× Cortex-A76**, so sweep the intra-op thread count and keep the **best** — ORT's default can over- or under-subscribe on the Pi.

```bash
source .venv/bin/activate
cd ~/deepweeds-edge-classifier

# fp32 — thread sweep
for t in 1 2 4; do
  echo "=== fp32  threads=$t ==="
  python benchmark.py --onnx runs/mnv3/model.onnx --runs 300 --warmup 30 --threads $t
done

# INT8 dynamic — same sweep
for t in 1 2 4; do
  echo "=== int8  threads=$t ==="
  python benchmark.py --onnx runs/mnv3/model.int8.onnx --runs 300 --warmup 30 --threads $t
done
```
Notes:
- **`--runs 300 --warmup 30`** (up from the defaults) — the first calls JIT/allocate; the longer warmup + larger sample tightens p95 on the Pi.
- Re-check `vcgencmd get_throttled` **after** the runs too — a long sweep can heat-soak. If it flipped off `0x0`, the later runs are suspect; cool down and redo.
- `benchmark.py` feeds random input (synthetic) — fine for **latency/throughput**. Accuracy is unchanged from the Mac eval (same weights); don't re-measure accuracy here.

## Phase 5 — Record the result

Take the **best thread count** for each model and add a Pi 5 row pair to the README results table, matching the existing format:

| Model | Latency (Pi 5 CPU, ONNX Runtime) | Throughput | Best threads |
|---|---|---|---|
| MobileNetV3-Large (fp32) | _mean ms (p50 / p95)_ | _FPS_ | _n_ |
| MobileNetV3-Large (INT8 dynamic) | _mean ms (p50 / p95)_ | _FPS_ | _n_ |

Then write the one-line finding, e.g.:
> On the Pi 5 (Cortex-A76, governor=performance, no throttling), fp32 runs at **X FPS**; INT8 dynamic is **slower / faster** at **Y FPS** — [confirming / contradicting] the Apple-Silicon result. This sets the **CPU baseline** the Hailo AI HAT+ is measured against.

Capture the conditions alongside the numbers (governor, `get_throttled=0x0`, peak temp, ORT version, OS) — that's what makes the eventual fp32-CPU → Hailo speed-up credible rather than hand-wavy.

---

## What this is *not*
- **Not the Hailo benchmark.** The Hailo AI HAT+ doesn't run through ORT's CPU provider — it compiles the model to a `.hef` and runs via HailoRT. That's a separate runbook, triggered when the HAT+ arrives.
- **Not an accuracy run.** Accuracy is dataset-dependent and already measured; this is pure on-device speed.

## Troubleshooting
- `Illegal instruction` / no wheel found → you're on a 32-bit OS or wrong arch; reflash 64-bit (`uname -m` must be `aarch64`).
- Latency creeping upward across a run → thermal throttling; check `vcgencmd measure_temp` / `get_throttled`, confirm the Active Cooler is seated and running.
- Wildly variable p95 → governor still `ondemand`; redo Phase 2 step 1.
- INT8 *much* slower than fp32 → **expected** on ARM CPU (no fast int8 depthwise kernels) — that's the finding, not a bug.
