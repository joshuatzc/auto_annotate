# auto_annotate

Automatically creates ELAN annotation files (`.eaf`) for TUG (Timed Up and Go)
and CRT (Chair Rise Test) videos. Place your test videos in the `input/` folder,
run one command, and get ready-to-review annotation files in the `output/` folder.

---

## Before you start (one-time setup)

You only need to do this once. You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
(or Anaconda) installed first.

**Mac / Linux** — open a terminal in this folder and run:
```bash
bash install.sh
conda activate auto_anno
```

**Windows** — open an Anaconda Prompt in this folder and run:
```
install.bat
conda activate auto_anno
```

After setup your terminal prompt will show `(auto_anno)`. This means the tool
is ready. You will need to run `conda activate auto_anno` again each time you
open a new terminal window.

---

## Everyday workflow

There are four steps. You run them in order, and you can repeat any step as many
times as you like.

---

### Step 1 — Add your test videos

Put your TUG or CRT test folders inside the `input/` folder. Each test folder
must contain at minimum a video file named `rgb_video_*.mp4`.

**Example layout:**
```
input/
  Patient001/
    Patient001 TUG/
      rgb_video_Patient001.mp4
      bounding_box_data_Patient001.json   ← optional, improves quality
  Patient002/
    Patient002 CRT/
      rgb_video_Patient002.mp4
```

The tool automatically detects whether a folder is TUG or CRT from the folder
name. See `input/README.txt` for the full list of optional files that improve
annotation quality.

---

### Step 2 — Run the annotator

**Mac / Linux:**
```bash
python3 -m auto_annotate.cli
```

**Windows:**
```
python -m auto_annotate.cli
```

The tool scans everything inside `input/`, creates an annotation file for each
test folder it finds, and saves the results in `output/`. The output folder
structure mirrors the input:

```
input/Patient001/Patient001 TUG/   →   output/Patient001/Patient001_TUG_auto.eaf
input/Patient002/Patient002 CRT/   →   output/Patient002/Patient002_CRT_auto.eaf
```

---

### Step 3 — Review each annotation in ELAN

**Mac / Linux:**
```bash
python3 -m auto_annotate.cli review -i input
```

**Windows:**
```
python -m auto_annotate.cli review -i input
```

This opens each `.eaf` file in ELAN one at a time. ELAN will load the matching
video automatically — you do not need to locate it yourself.

**Controls:**

| Key | What it does |
|-----|--------------|
| `o` | Open the current file in ELAN |
| Enter or `d` | Mark as done → move to next file |
| `c` | Mark as corrected → move to next file (use this after saving your changes in ELAN) |
| `s` | Skip for now → move to next file |
| `n` | Go to the next file without changing its status |
| `p` | Go back to the previous file |
| `g` | Jump to a specific file number |
| `q` | Quit — your progress is always saved automatically |

**How to correct an annotation:**
1. Press `o` to open the file in ELAN.
2. Make your changes in ELAN.
3. Save in ELAN — **Cmd+S** on Mac or **Ctrl+S** on Windows.
4. Switch back to the terminal and press `c`.

You can stop and resume the review at any time. Progress is saved to
`output/.review_state.json` after every key press.

---

### Step 4 — Improve future annotations from your corrections

After you have corrected several files, run the calibrator. It reads your
corrections and works out better settings for the annotator.

**Mac / Linux:**
```bash
python3 -m auto_annotate.cli calibrate
```

**Windows:**
```
python -m auto_annotate.cli calibrate
```

This shows you what would change. To apply the changes immediately:

**Mac / Linux:**
```bash
python3 -m auto_annotate.cli calibrate --apply
```

**Windows:**
```
python -m auto_annotate.cli calibrate --apply
```

Then go back to **Step 2** and re-run the annotator. The new annotation files
will be more accurate.

> **Tip:** The more corrections you feed in, the better the results. Aim for at
> least 5–10 corrected files before calibrating.

---

## Troubleshooting

**ELAN asks me to locate the video file**

This means the annotation file was created on a different computer or the folders
have been moved. Run the review command with `-i input` (as shown in Step 3) —
it will repair all the video links automatically before you start reviewing.

**`python3: command not found` or `ModuleNotFoundError: No module named 'cv2'`**

The conda environment is not active. Run:
```bash
conda activate auto_anno
```

**`(auto_anno)` is not shown in my terminal prompt**

Same fix — run `conda activate auto_anno` before using any commands.

**"No corrected files found" when running calibrate**

You need to use the `c` key during review (not just `d`) to flag files that you
have corrected in ELAN. Only files marked with `c` are used for calibration.

**"No usable calibration data" when running calibrate**

Calibration only improves the fallback annotator (used when no bounding box data
is available). If all your files were annotated using detected bounding boxes, the
calibration has nothing to tune. This is expected and means your annotations are
already using the best available method.

---

## Reference

### All commands

| Command | What it does |
|---------|--------------|
| `python3 -m auto_annotate.cli` | Annotate everything in `input/`, save to `output/` |
| `python3 -m auto_annotate.cli review -i input` | Review output files in ELAN one by one |
| `python3 -m auto_annotate.cli calibrate` | Show suggested improvements from your corrections |
| `python3 -m auto_annotate.cli calibrate --apply` | Apply those improvements to the annotator |
| `python3 -m auto_annotate.cli -i FOLDER -o FILE.eaf` | Annotate a single folder |
| `python3 -m auto_annotate.cli batch INPUT OUTPUT` | Annotate a custom input root into a custom output folder |

On Windows, replace `python3` with `python` in all commands above.

### Annotation confidence levels

The tool uses different methods depending on what data is available. Higher
confidence means less likely to need correction.

| Confidence | Method | What data was used |
|------------|--------|--------------------|
| High (0.95) | Precomputed pipeline output | `out2.pkl`, `out2.csv`, or `sppb_results.json` |
| High (0.95) | Face + body detection | `bounding_box_face_data_*.json` |
| Medium (0.70) | Body motion analysis | `bounding_box_data_*.json` only |
| Low (0.45) | Proportional estimate | Video duration only (no detection data) |
| None (0.00) | Unknown | Insufficient input data |

Calibration (Step 4) improves the **medium and low confidence** cases.
