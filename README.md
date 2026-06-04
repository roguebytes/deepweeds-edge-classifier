# DeepWeeds Edge Classifier

Train an image classifier on the **DeepWeeds** Australian rangeland-weed dataset, then export and benchmark it on a low-cost edge device (Raspberry Pi / Jetson). The focus isn't just accuracy — it's **on-device inference**: getting a model to run efficiently at the edge.

> **Scope:** DeepWeeds is *whole-image classification* of *ground-level* photos — not aerial imagery and not object detection. This repo demonstrates image classification + edge deployment on a real Australian agricultural dataset.

## Dataset

DeepWeeds: 17,509 images, 8 weed species + 1 negative (non-weed) class, collected in situ across 8 northern-Australia rangeland sites (Olsen et al., 2019). Images are 256×256 px. Get it from the official release (GitHub `AlexOlsen/DeepWeeds`), TensorFlow Datasets (`deep_weeds`), or Hugging Face, then arrange as:

```
data/
  images/           # all .jpg files
  labels.csv        # columns: Filename, Label, Species   (Label = 0..8)
```

Label map (index → species): `0 Chinee apple, 1 Lantana, 2 Parkinsonia, 3 Parthenium, 4 Prickly acacia, 5 Rubber vine, 6 Siam weed, 7 Snake weed, 8 Negative`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # see note on installing torch for your platform
```

## Usage

```bash
# 1. Train (auto-detects CUDA > MPS > CPU). ResNet-50 baseline:
python train.py --data-dir data/images --labels-csv data/labels.csv \
    --arch resnet50 --epochs 15 --batch-size 32 --output-dir runs/resnet50

# Lightweight model you'll actually deploy to the edge:
python train.py --data-dir data/images --labels-csv data/labels.csv \
    --arch mobilenet_v3_large --epochs 20 --output-dir runs/mnv3

# 2. Export the trained model to ONNX (optionally INT8-quantized):
python export.py --checkpoint runs/mnv3/best_model.pt --arch mobilenet_v3_large \
    --output runs/mnv3/model.onnx --quantize

# 3. Benchmark inference latency (run this on the Pi/Jetson too):
python benchmark.py --onnx runs/mnv3/model.onnx --runs 200
```

Outputs land in `--output-dir`: `best_model.pt`, `metrics.json`, `confusion_matrix.png`.

### Smoke test (no dataset/network needed)
Verify the whole train → export → benchmark loop wires together on synthetic data:
```bash
python smoke_test.py
```
It generates a tiny fake dataset, trains 1 epoch with `--no-pretrained`, exports to ONNX (incl. INT8), and benchmarks — then prints `SMOKE TEST PASSED`. Requires the deps in `requirements.txt`.

### Installing PyTorch
`requirements.txt` lists `torch`/`torchvision` loosely. For CUDA or a specific build, install from the official selector at pytorch.org first, then `pip install -r requirements.txt`.

## Results

Trained on Apple-Silicon MPS, 20 epochs, inverse-frequency class weighting; evaluated on a held-out test split.

| Model | Params | Test acc | Size | Latency (Mac CPU, ONNX Runtime) | Throughput |
|---|---|---|---|---|---|
| MobileNetV3-Large (fp32) | 4.2 M | **97.15 %** | 16 MB | 5.83 ms (p50 4.88, p95 8.22) | 171.6 FPS |
| MobileNetV3-Large (INT8 dynamic) | 4.2 M | — | **4.2 MB** | 39.56 ms | 25.3 FPS |

**On-device — Raspberry Pi 5** (Cortex-A76 ×4, ONNX Runtime CPU; best of a 1/2/4-thread sweep, 300 runs after 30 warmup, `governor=performance`, no throttling):

| Model | Precision | Latency (mean / p95) | Throughput | Threads |
|---|---|---|---|---|
| MobileNetV3-Large | fp32 | 14.23 ms / 14.54 ms | **70.3 FPS** | 4 |
| MobileNetV3-Large | INT8 dynamic | 68.33 ms / 68.55 ms | 14.6 FPS | 4 |

**Findings**
- The lightweight MobileNetV3-Large (4.2 M params) reaches **97.15 %** — above the dataset paper's ResNet-50 baseline of **95.7 %** (Olsen et al., 2019), with a model ~6× smaller.
- **Balanced per-class recall (0.95–0.99)** despite the Negative class being ~half the data — inverse-frequency class weighting (`--class-weights`) stops the majority class dominating. Weakest is Snake weed (0.95), mostly confused with Chinee apple (see confusion matrix).
- **INT8 dynamic quantization shrinks the model ~3.8× (16 → 4.2 MB) but is *slower* on this ARM CPU** (39.6 vs 5.8 ms): the per-op quantize/dequantize overhead outweighs int8 compute for MobileNet's depthwise convs, and ONNX Runtime's CPU provider lacks fast int8 kernels here. **fp32 is the CPU deployment choice; INT8's speed win needs an accelerator (e.g. Hailo) or static quantization on VNNI-class x86.**
- **On the deployment target (Pi 5, Cortex-A76), fp32 reaches 70.3 FPS while INT8 dynamic is ~4.8× *slower* (14.6 FPS)** — the same ARM-CPU effect seen on the Mac, now confirmed on real hardware (run-to-run std < 0.2 ms at `governor=performance`, no throttling). See the [Pi 5 CPU benchmark runbook](docs/pi5_benchmark.md). This fp32 result is the **CPU baseline** the Hailo accelerator is measured against.
- The INT8 speed-up that the CPU can't deliver is the job of a dedicated NPU — see the [Hailo AI HAT+ setup + benchmark runbook](docs/hailo_benchmark.md) for fitting the accelerator, compiling this model to a Hailo `.hef`, and the three-way CPU-vs-NPU results table.

![Confusion matrix](docs/confusion_matrix.png)

## Reference
Olsen, A., Konovalov, D. A., Philippa, B., et al. (2019). DeepWeeds: A multiclass weed species image dataset for deep learning. *Scientific Reports, 9*(1), 2058. https://doi.org/10.1038/s41598-018-38343-3
