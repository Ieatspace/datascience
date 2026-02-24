# Handwrite Studio (Letter Model Phase 1)

This project renders handwritten PNG previews in a Next.js app.

Phase 1 adds a lightweight letter-level learned generator (`python_ai/`) that learns per-letter variation from your scanned/labeled glyph crops and plugs into the existing `/api/generate` flow without changing the frontend preview contract.

## Current Website Flow

- Frontend calls `POST /api/generate`
- Next.js route validates request with Zod
- `lib/generate-service.ts` runs local Python `generate_handwriting_page.py` (or FastAPI/stub)
- Python returns a PNG file, Next.js converts it to `imageDataUrl` (`data:image/png;base64,...`)
- `PreviewPanel` renders the PNG unchanged

## Dataset Preparation (Phase 1 Letter Model)

The letter model trains from your existing lowercase character dataset:

- `out/labels.csv`
  - CSV with at least: `filename,label`
  - `label` should be lowercase `a-z`
- `out/chars/`
  - PNG crops matching filenames in `out/labels.csv`

Notes:

- Input is scanned-paper-derived glyph crops (no stroke/tablet data required).
- The model currently targets lowercase `a-z`.
- Spaces are handled by the renderer (not the model).
- Missing letters fall back to the existing crop sampler or fallback glyph path.

## Phase 1: Train the Letter Model

Recommended (CPU-friendly defaults):

```bash
python -m python_ai.train --epochs 20 --batch-size 64
```

Useful options:

```bash
python -m python_ai.train ^
  --epochs 30 ^
  --batch-size 64 ^
  --image-size 64 ^
  --latent-dim 32 ^
  --beta 0.15 ^
  --val-split 0.12 ^
  --seed 1234 ^
  --checkpoint-dir out/checkpoints ^
  --out-weights out/letter_gen.pt ^
  --out-config out/letter_gen.json
```

Outputs:

- Weights: `out/letter_gen.pt`
- Training config/summary JSON: `out/letter_gen.json`
- Epoch checkpoints: `out/checkpoints/letter_gen_epochXXX.pt`

## Sanity Check: Sample Generated Letters

Generate sample grid for one letter:

```bash
python -m python_ai.sample --letter a
```

Generate one sample for every letter:

```bash
python -m python_ai.sample --letter all
```

Sample grids are written to `out/generated/` by default.

## Smoke Test (Python)

This script:

- trains for 1 epoch on a small subset
- then renders `"hello world"` via `generate_handwriting_page.py` using the learned letter model

```bash
python -m python_ai.smoke_test
```

## Website Integration (Automatic)

The local Python provider now auto-enables the learned letter model when:

- `HANDWRITE_PROVIDER` resolves to local mode, and
- `out/letter_gen.pt` exists (or `HANDWRITE_LOCAL_LETTER_MODEL_PATH` points to a weights file)

If the letter model is unavailable or fails:

- it falls back to the existing crop sampler (`out/chars` + `out/labels.csv`)
- if that also fails for a character, it falls back to the existing synthetic fallback glyph path

The frontend `Generate Preview` flow and response shape remain compatible.

## Run the Website and Generate an Image

Start Next.js:

```bash
npm run dev
```

Open the app and click `Generate Preview`, or use the local smoke script:

```bash
npm run smoke:generate
```

The smoke script posts to `http://localhost:3000/api/generate` and asserts the response starts with:

- `data:image/png;base64,`

## Local Generator Tuning (Optional)

The request schema supports optional hidden fields (not required by the UI) for letter-model tuning:

- `letterModelEnabled`
- `letterModelStyleStrength`
- `letterModelBaselineJitter`
- `letterModelWordSlant`
- `letterModelRotationJitter`
- `letterModelInkVariation`

The renderer also applies realism tweaks with defaults:

- baseline jitter (correlated within words)
- pairwise kerning adjustments
- per-word slant
- per-letter rotation jitter
- light ink thickness variation

## Environment Variables (Local Provider)

See `.env.example`. Relevant additions:

- `HANDWRITE_LOCAL_USE_LETTER_MODEL=1`
- `HANDWRITE_LOCAL_LETTER_MODEL_PATH=out/letter_gen.pt`
- `HANDWRITE_LOCAL_USE_CLASSIFIER=1`

## Troubleshooting

### Missing letters / weak quality for some letters

- Check class counts in `out/labels.csv`
- Add more samples for the weak letters
- Retrain (`python -m python_ai.train ...`)
- The renderer will fall back to crop sampling or fallback glyphs for unsupported/low-sample letters

### Blurry outputs

- Increase training epochs
- Reduce `--beta` slightly (e.g. `0.10`)
- Keep `--image-size` at `64` (higher sizes require more data/training)
- Verify crop labels are clean and not heavily clipped

### Letter model not being used in the website

- Confirm `out/letter_gen.pt` exists
- Check `.env` values (`HANDWRITE_PROVIDER`, `HANDWRITE_LOCAL_USE_LETTER_MODEL`)
- Look for warnings in server logs; the generator falls back instead of crashing

### CPU training is slow

- Use fewer epochs first (`--epochs 5`) to validate the pipeline
- Use `--max-samples` for iteration
- Reduce `--base-channels` or `--latent-dim` for quicker experiments

## Phase 2: Full Sentence Model (Scaffold Only)

Phase 2 is intentionally scaffolded and not implemented yet.

Added placeholders:

- `python_ai/text2handwriting/dataset_line.py`
- `python_ai/text2handwriting/model_line.py`
- `python_ai/text2handwriting/train_line.py`

### What Phase 2 will need

Paired data:

- one text label
- one handwritten line image
- exact alignment (the line image must contain exactly that text)

### Recommended collection method

- Use prompt sheets
- Write one line per prompt
- Save each line as a separate cropped image
- Store labels in CSV/JSONL (`split,text,image_path`)

### Expected future dataset format

- `train.csv` / `val.csv` (or JSONL)
- columns: `text,image_path`
- images pre-cropped to a single handwritten line

Phase 2 training script currently raises `NotImplementedError` with guidance so the repo has a clean extension point without pretending a large text-conditioned model is ready.
