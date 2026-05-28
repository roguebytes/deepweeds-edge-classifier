# Getting Started (PyCharm) — DeepWeeds Edge Classifier

A step-by-step runbook from a fresh checkout to a published result. Examples use PyCharm on macOS (Apple Silicon → training uses **MPS**), but the underlying commands work in any environment. Menu paths are for recent PyCharm (2024/2025); Community Edition is fine.

---

## Phase A — Project, interpreter, deps, sanity run (~30 min, mostly the torch download)

**Before you start — confirm a base Python.** PyCharm builds the venv *from* an existing Python. In a terminal: `python3 --version`. You want **3.11 or 3.12** (Apple-Silicon/arm64). If missing, install from python.org (arm64 installer) or `brew install python@3.12`. (Avoid 3.13 for now — some wheels still lag.)

1. **Open the project.** `File → Open…` → select the `deepweeds-edge-classifier` folder (the repo root itself). If prompted, **Trust** the project. Opening the wrong folder is the #1 cause of later `No module named 'deepweeds'` errors.

2. **Create the project venv (interpreter).**
   - `PyCharm → Settings` (⌘,) → `Project: deepweeds-edge-classifier → Python Interpreter`.
   - Click **Add Interpreter** (top-right of the pane) → **Add Local Interpreter…**.
   - Left list: **Virtualenv Environment** → select **New** (not "Existing").
   - **Base interpreter:** pick your Python 3.12. **Location:** leave the default `…/deepweeds-edge-classifier/.venv`. Click **OK**.
   - *(Older PyCharm: the ⚙ gear icon → Add → Virtualenv Environment → New environment.)*
   - **Confirm:** the bottom-right status bar reads `Python 3.12 (deepweeds-edge-classifier)`.

3. **Install dependencies.**
   - PyCharm usually shows a banner *"Package requirements … are not satisfied"* → click **Install requirements**.
   - No banner? Use the built-in **Terminal** (`⌥F12`, runs inside `.venv`): `pip install -r requirements.txt` — or the **Python Packages** tool window (`View → Tool Windows → Python Packages`).
   - **This downloads PyTorch (~hundreds of MB) — allow 3–10 min.** The default Apple-Silicon wheel includes **MPS** (Metal GPU) support; nothing extra needed.
   - **Verify it:** open the **Python Console** (`View → Tool Windows → Python Console`) and run `import torch; print(torch.__version__, torch.backends.mps.is_available())` — expect a version string and **`True`** (`False` just means CPU fallback — slower but fine).

4. **Sanity-run the pipeline.** In the Project tool window, right-click **`smoke_test.py` → Run 'smoke_test'**. The **Run** tool window generates a tiny synthetic dataset, trains 1 epoch, exports ONNX (incl. INT8), benchmarks, and ends with **`SMOKE TEST PASSED`** (~1–2 min). That proves the whole train→export→benchmark chain works *before* you download 17k images.

   **If it fails:**
   - `ModuleNotFoundError: torch` → interpreter isn't the `.venv`, or deps didn't install (recheck 2–3; look at the status-bar interpreter).
   - `No module named 'deepweeds'` → wrong project root; reopen `deepweeds-edge-classifier` itself (step 1).
   - Otherwise → inspect the Run-window traceback against the steps above.

---

## Phase B — Get the dataset (built-in Terminal, ~20 min, 468 MB)

Open the Terminal (`View → Tool Windows → Terminal`, or ⌥F12). It runs inside `.venv`. Then:

```bash
pip install gdown
gdown 1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj -O images.zip      # 468 MB
unzip -q images.zip -d data/                                # → data/images/*.jpg
git clone --depth 1 https://github.com/AlexOlsen/DeepWeeds /tmp/deepweeds-src
cp /tmp/deepweeds-src/labels/labels.csv data/labels.csv
```

Verify: `ls data/images | wc -l` → ~**17509**; `head -3 data/labels.csv` → `Filename,Label,Species` (Label 0–8; **8 = Negative**, which is ~half the data — that's why we use `--class-weights` below).

**Keep PyCharm fast:** right-click the `data/` folder → `Mark Directory as → Excluded` (don't let it index 17k images). Do the same for `runs/` once it exists.

---

## Phase C — Train the deployable model (Run Configurations)

Create a Run Configuration: `Run → Edit Configurations… → + → Python`.
- **Name:** `train-mnv3`
- **Script:** `train.py`
- **Parameters:** `--data-dir data/images --labels-csv data/labels.csv --arch mobilenet_v3_large --epochs 20 --batch-size 32 --class-weights --output-dir runs/mnv3`
- **Working directory:** the repo root (so `data/...` resolves)
- **Python interpreter:** your `.venv`

5. **Quick sanity first:** duplicate that config (`Edit Configurations → Copy`), name it `train-mnv3-sanity`, change Parameters to `--epochs 1 --output-dir runs/mnv3-sanity` (keep the rest). Run it. If you hit a macOS dataloader/multiprocessing error, add `--num-workers 0` to Parameters.

6. **Full train:** select `train-mnv3` in the toolbar dropdown → Run (▶). Watch `val acc` climb in the Run tool window (a few min/epoch on MPS). Outputs land in `runs/mnv3/`: `best_model.pt`, `metrics.json`, `confusion_matrix.png`.

> Adjust hyper-params anytime via the config dropdown → `Edit Configurations…` — no need to retype commands.

---

## Phase D — Reproduce the paper baseline (ResNet-50)

7. Duplicate the config → `train-resnet50`, Parameters: `--data-dir data/images --labels-csv data/labels.csv --arch resnet50 --epochs 15 --batch-size 32 --output-dir runs/resnet50`. Target ≈ **95%** (published baseline 95.7%) — a direct comparison against the paper, where MobileNetV3 is the smaller, deployable counterpart.

---

## Phase E — Export + edge benchmark (Run Configurations)

8. **Export config:** `+ → Python`, name `export-mnv3`, Script `export.py`, Parameters `--checkpoint runs/mnv3/best_model.pt --arch mobilenet_v3_large --output runs/mnv3/model.onnx --quantize`. Run.

9. **Benchmark config:** name `benchmark-mnv3`, Script `benchmark.py`, Parameters `--onnx runs/mnv3/model.onnx --runs 200`. Run → read mean ms/frame + FPS in the Run window. The same script can be re-run on a Raspberry Pi 5 for an on-device number. (A Hailo accelerator is a later optimisation needing the Hailo toolchain; ONNX-Runtime-CPU is the simple cross-platform baseline.)

---

## Phase F — Publish (PyCharm Git integration)

10. View results: double-click `runs/mnv3/confusion_matrix.png` to open it in the editor; open `metrics.json` for accuracy + per-class report. Fill the README results table (model · params · accuracy · edge latency) and keep the scope note (ground-level classification, not aerial).
11. Commit + push from PyCharm:
    - `VCS → Enable Version Control Integration… → Git` (if not already a repo).
    - **Commit** tool window (left edge, or ⌘0): stage files, write a message, Commit. The repo's `.gitignore` already excludes `data/`, `runs/`, `*.zip`, `*.pt`, `*.onnx`.
    - `Git → GitHub → Share Project on GitHub` (sign in under `Settings → Version Control → GitHub`), then `Git → Push` (⌘⇧K).

---

## PyCharm tips / troubleshooting
- **Live logs:** the Run tool window streams the per-epoch `train/val acc` prints.
- **MPS out of memory:** lower `--batch-size` to 16 in the Run Configuration.
- **Dataloader hang/error on macOS:** add `--num-workers 0` to Parameters.
- **First run downloads ImageNet weights** (~100 MB for ResNet-50) — needs network once.
- **Keep indexing fast:** mark `data/` and `runs/` as *Excluded*.
- **Match the paper's exact splits:** the cloned repo has 5-fold split CSVs in `/tmp/deepweeds-src/labels/`; pass them via `--train-csv/--val-csv/--test-csv`. Otherwise the script does a deterministic stratified split of `labels.csv`.
