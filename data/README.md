# Where Paper 03's data actually lives

This folder is intentionally (almost) empty. Paper 03 **reuses Paper 01's converted
ScenarioNet databases in place** — no copies, no new downloads (see `DOWNLOAD.md`). Copying
would duplicate ~tens of GB and let the two papers silently drift onto different splits.

All paths are defined once, in `src/simenv.py` → `DBS`:

| key | role in Paper 03 | path | scenarios |
|---|---|---|---|
| `av2_dev` | **development** — design choices (harm definition, score form, λ grid) are made here | `F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_splits\val\train` | 14 990 |
| `av2_cal` | **calibration** — LTT / CRC threshold selection | `...\data\av2_splits\val\cal` | 4 971 |
| `av2_test` | **test** — touched only by pre-registered evaluations | `...\data\av2_splits\val\test` | 5 027 |
| `ns_val0` | nuScenes (shift, H3) — prediction-challenge snippets, ~2.5 s each | `...\data\nuscenes_scenarionet\val\val_0` | part of 9 041 |

The three AV2 splits are deterministic scenario-ID-hash splits of AV2 **val** made by
Paper 01's `src/split_db.py` (salt `av2v1`). AV2 val is disjoint from AV2 train, on which
Paper 01's AutoBot was trained, so none of these scenarios leaked into the predictor.

Raw sources (Paper 01): AV2 Motion Forecasting in `...\data\argoverse2\`, nuScenes
metadata + maps in `...\data\nuscenes\v1.0-trainval_meta\`.

## Still needed (no download — conversion only)

- **nuScenes full logs** (`v1.0-trainval`, ~20 s scenes) for closed-loop H3. The metadata
  and maps are already on disk; sensor blobs are not required.
