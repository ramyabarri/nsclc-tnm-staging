# Running nnU-Net training on Vast.ai — step-by-step guide

This walks you through renting a GPU on Vast.ai and training nnU-Net v2 on the
NSCLC-Radiomics cohort using `notebooks/experiments/02_nnunet_vastai_training.ipynb`.

**What you'll produce:** a trained nnU-Net GTV segmentation model
(`nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres`, fold 0) zipped for download.

**Time & cost:** ~3–4.5 h wall-clock end to end; a few USD on a single RTX 4090
(~$0.30–0.50/hr). The GPU-heavy part (training) is ~1.5–3 h.

> ⚠️ **Billing rule of thumb:** Vast.ai charges for every hour an instance is
> *rented*, whether or not you're using it. **Destroy the instance the moment
> you've downloaded your model.** See Phase 7.

---

## Phase 0 — Prerequisites (do this before renting)

1. **Vast.ai account + credit.** Sign up at https://vast.ai, add a small amount of
   credit (Billing tab). $10 is plenty for one training run.
2. **Two files to upload later** (have them ready on your laptop):
   - `notebooks/experiments/02_nnunet_vastai_training.ipynb`
   - `scripts/preprocessing/prepare_nnunet_dataset.py`
3. **No TCIA login needed.** NSCLC-Radiomics is a public collection, so the
   `tcia_utils` download in the notebook works without credentials.

---

## Phase 1 — Rent a GPU instance

1. In the Vast.ai console, go to **Search** (a.k.a. **Create / Rent**).
2. **Pick the Docker image / template.** Click **"Edit Image & Config"** and choose a
   **PyTorch** template (e.g. `pytorch/pytorch` with CUDA, or Vast's "PyTorch" recommended
   image). This gives you CUDA + PyTorch + Jupyter preinstalled, so `nnunetv2` won't have
   to pull a fresh torch build.
3. **Set disk size to ≥ 100 GB** using the disk slider *before* renting (raw DICOM +
   NIfTI + nnU-Net preprocessed data need the room). Setting it now is easier than
   resizing later.
4. **Filter the GPU list:**
   - GPU: **RTX 4090** or **RTX 3090** (24 GB) — cheap and more than enough. A100 works
     but costs more.
   - **On-demand**, not interruptible, for a training run (interruptible instances can be
     preempted mid-training; on-demand won't be). If you do use interruptible, the
     notebook's resume cell handles restarts.
   - Sort by **$/hr** and pick a well-reviewed host with good reliability and bandwidth
     (download speed matters — you're pulling ~40 GB from TCIA).
5. Click **Rent**. The instance appears under the **Instances** tab and takes a minute
   or two to boot (status → "running").

---

## Phase 2 — Open Jupyter and upload the files

1. On the **Instances** tab, click **"Open"** / the **Jupyter** button on your instance.
   (Vast shows a Jupyter URL with a token; the PyTorch image runs Jupyter on the mapped
   port automatically.)
2. In Jupyter, navigate to **`/workspace`** (the notebook expects this path).
3. **Upload both files** with the Jupyter **Upload** button (top-right of the file
   browser):
   - `02_nnunet_vastai_training.ipynb`
   - `prepare_nnunet_dataset.py`  ← must sit in the **same directory** you run the
     notebook from (the notebook checks for it in section 4).
4. Open the notebook.

> **SSH alternative:** if you prefer the terminal, use the SSH command Vast shows on the
> instance card, then `scp` the two files to `/workspace/`. Jupyter upload is simpler.

---

## Phase 3 — Run the setup cells (sections 1–4)

Run top to bottom. Expected results:

| Section | What it does | Expect |
|---|---|---|
| **1** Check GPU & disk | `nvidia-smi` + disk assert | Prints your GPU name; asserts ≥ 100 GB free. If it fails here, your disk is too small — destroy and re-rent with more disk. |
| **2** Install deps | `pip install nnunetv2 pydicom rt_utils SimpleITK tcia_utils …` | ~2–3 min, ends with `Done`. |
| **3** Configure paths | Sets `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` env vars + `TRAINER` | Prints the three paths and `TRAINER = nnUNetTrainer_250epochs`. |
| **4** Check script | Asserts `prepare_nnunet_dataset.py` is present | Prints `Found …/prepare_nnunet_dataset.py`. If it fails, re-upload the script into this directory. |

---

## Phase 4 — Download + convert the data (sections 5–6, ~75 min)

- **Section 5 (download, ~45 min):** fetches the CT + RTSTRUCT series list from TCIA, then
  downloads them into `/workspace/data/raw`. The last cell asserts DICOMs were downloaded.
  This is the longest wall-clock step and depends on the host's bandwidth.
- **Section 6 (convert, ~30 min):** runs `prepare_nnunet_dataset.py`, which reads each
  series' DICOM header, groups by PatientID, and writes nnU-Net format
  (`imagesTr/*_0000.nii.gz` + `labelsTr/*.nii.gz` + `dataset.json`). The verify cell should
  print roughly **422 images / 422 labels** and the `dataset.json` contents
  (`numTraining` ≈ 422). A few patients skipping for "no RTSTRUCT" is fine.

> If image/label counts are 0, scroll up in the conversion output — it logs per-patient
> `OK` / `FAILED` and writes `_conversion_failures.json`.

---

## Phase 5 — Plan, preprocess, and train (sections 7–8)

- **Section 7 (plan & preprocess, ~25 min):** `nnUNetv2_plan_and_preprocess -d 001
  --verify_dataset_integrity -c 3d_fullres`. nnU-Net fingerprints the dataset and
  auto-configures the network. `--verify_dataset_integrity` will flag any CT/mask
  geometry mismatch here — if it errors, stop and check the conversion, don't train on
  bad data.
- **Section 8 (train, ~1.5–3 h):** `nnUNetv2_train 001 3d_fullres 0 -tr
  nnUNetTrainer_250epochs --val_best -device cuda`. This is the GPU-heavy part. It prints
  per-epoch loss and pseudo-Dice. **You can close the laptop** — the run continues on the
  instance. Checkpoints are written every 50 epochs.

**If your session disconnects** (browser closed, laptop slept, interruptible preemption):
reopen Jupyter, re-run sections 1 and 3 (to reinstate env vars), then run the **resume
cell** under section 8 (uncomment it) — it adds `--c` to continue from the latest
checkpoint. Don't re-run download/convert; that data persists on the instance's disk.

---

## Phase 6 — Inspect and package (sections 9–10)

- **Section 9:** prints the fold-0 validation Dice from `summary.json`. For NSCLC GTV on
  this cohort, expect a real, non-trivial DSC (roughly the 0.6–0.8 range — this is your
  actual segmentation baseline, replacing the zero-mask fallback).
- **Section 10:** zips the model to `/workspace/nnunet_model_Dataset001_NSCLCRadiomics.zip`
  and prints its size and an `scp` command.

**Download the zip** via the Jupyter file browser (right-click → Download), or the `scp`
command it prints. It's typically a few hundred MB.

---

## Phase 7 — ⚠️ Destroy the instance

Once the zip is safely on your laptop:

1. Go to the **Instances** tab.
2. Click **Destroy** (the trash icon) on your instance — not just "Stop". Stopping keeps
   billing you for storage; **Destroy** ends all charges.
3. Confirm the instance is gone.

Double-check your **Billing** tab afterward to confirm no instance is still running.

---

## Phase 8 — Local setup after download

On your laptop, in the project root:

```bash
mkdir -p results/nnunet
unzip nnunet_model_Dataset001_NSCLCRadiomics.zip -d results/nnunet/
```

This gives you:

```
results/nnunet/Dataset001_NSCLCRadiomics/
  nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/
    dataset.json  plans.json  fold_0/checkpoint_best.pth
```

which is exactly the default `--nnunet-model` path in `run_baselines.py`. Re-run the
sanity baseline to get real nnU-Net numbers (in the `nsclc-mas` env):

```bash
python -m scripts.evaluation.run_baselines
```

The `nnunet` row in `results/segmentation/week5_baselines.csv` should now show a real
Dice instead of 0.000.

---

## Cheatsheet

| Thing | Value |
|---|---|
| GPU | RTX 4090 / 3090 (24 GB), on-demand |
| Disk | ≥ 100 GB, set at rent time |
| Image | PyTorch (CUDA + Jupyter preinstalled) |
| Working dir | `/workspace` |
| Files to upload | `02_nnunet_vastai_training.ipynb`, `prepare_nnunet_dataset.py` |
| TCIA login | none (public collection) |
| Trainer | `nnUNetTrainer_250epochs` (switch to `nnUNetTrainer` for full 1000-epoch) |
| Total time / cost | ~3–4.5 h / a few USD |
| When done | **Destroy** the instance |
