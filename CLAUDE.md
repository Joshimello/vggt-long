# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT-Long is a monocular 3D reconstruction system for kilometer-scale, unbounded outdoor environments. It extends foundation models (VGGT, Pi3, MapAnything) to long RGB sequences via chunk-based processing, overlapping alignment, and loop closure optimization.

## Environment Setup

```bash
conda create -n vggt-long python=3.10.18
conda activate vggt-long
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Download pretrained weights:
```bash
bash ./scripts/download_weights.sh
```

Optional C++ solver (faster, more stable loop closure):
```bash
python setup.py install
```

Optional DBoW2 (CPU-only VPR):
```bash
sudo apt-get install -y libopencv-dev
cd DBoW2 && mkdir -p build && cd build && cmake .. && make && sudo make install && cd ../..
pip install ./DPRetrieval
```

## Running

```bash
python vggt_long.py --image_dir ./path_of_images
python vggt_long.py --image_dir ./path_of_images --config ./configs/base_config.yaml
```

Extract frames from video first:
```bash
ffmpeg -i your_video.mp4 -vf "fps=5,scale=518:-1" ./extract_images/frame_%06d.png
```

Convert outputs to COLMAP format:
```bash
python convert_colmap.py  # see script args for options
```

## Architecture

### Main Pipeline (`vggt_long.py`)

The `VGGT_Long` class orchestrates the entire pipeline:

1. **Loop Detection** — Before model inference, detect loop closure pairs across all frames using VPR (Visual Place Recognition). Two backends: DINOv2+SALAD (GPU) or DBoW2 (CPU-only).
2. **Chunk Inference** — Process the image list in overlapping chunks. Each chunk result is saved to disk as `.npy` files (`_tmp_results_unaligned/`) rather than held in RAM to avoid CPU OOM.
3. **Sequential Alignment** — For each adjacent chunk pair, estimate a Sim(3) or SE(3) transform using the overlapping frames' world point maps via weighted IRLS.
4. **Loop Closure Optimization** — If loops are detected, run separate chunk inference on loop frame pairs, estimate the loop Sim(3) constraint, then globally optimize all chunk poses using Levenberg-Marquardt (via pypose).
5. **Apply Transforms & Save** — Accumulate and apply the aligned transforms to each chunk; save per-chunk colored point clouds as `.ply` to `pcd/`, and merge into `combined_pcd.ply`. Camera poses saved to `camera_poses.txt` and `camera_poses.ply`. Temp files auto-deleted.

### Model Abstraction (`base_models/`)

`Base3DModel` (ABC) defines the unified interface. All adapters must implement:
- `load()` — load weights into GPU
- `infer_chunk(image_paths) -> dict` — return dict with keys: `world_points`, `world_points_conf`, `extrinsic` (C2W 4×4), `intrinsic`, `depth`, `depth_conf`, `images`, `mask`

Adapters: `VGGTAdapter`, `Pi3Adapter`, `MapAnythingAdapter`. Select via `Weights.model` in config (`'VGGT'`, `'Pi3'`, `'Mapanything'`).

### Loop Closure (`LoopModels/`, `loop_utils/`)

- `LoopDetector` (`LoopModels/LoopModel.py`) — DINOv2+SALAD-based VPR; extracts per-image descriptors, searches with FAISS, applies NMS to prune false positives.
- `RetrievalDBOW` (`LoopModelDBoW/`) — DBoW2-based alternative; processes frames online, CPU-only.
- `Sim3LoopOptimizer` (`loop_utils/sim3loop.py`) — Levenberg-Marquardt Sim(3) pose graph optimizer using pypose. Uses either C++ solver (`sim3solve`, from `fastloop/solve.cpp`) or Python fallback (`fastloop/solve_python.py`). Falls back to Python automatically if C++ not compiled.

### Alignment Math (`loop_utils/sim3utils.py`)

- `weighted_align_point_maps` — confidence-weighted Sim(3)/SE(3) estimation between overlapping point maps (IRLS).
- `accumulate_sim3_transforms` — converts relative sequential transforms to absolute cumulative transforms.
- `apply_sim3_direct` — applies a Sim(3) to world point maps.
- `warmup_numba` — must be called before processing if `align_method: 'numba'` (default).

## Configuration (`configs/`)

All config files use YAML with `inherit_from` inheritance. The base is `configs/base_config.yaml`.

Key parameters:
| Key | Description |
|-----|-------------|
| `Weights.model` | Foundation model: `'VGGT'`, `'Pi3'`, `'Mapanything'` |
| `Model.chunk_size` | Frames per chunk (default: 60) |
| `Model.overlap` | Overlap between adjacent chunks (default: 30) |
| `Model.loop_enable` | Enable loop closure detection & optimization |
| `Model.useDBoW` | Use DBoW2 (CPU) vs SALAD/DINOv2 (GPU) for loop detection |
| `Model.using_sim3` | `True` = Sim(3) alignment (scale-free); `False` = SE(3) (metric, use with MapAnything) |
| `Model.align_method` | `'numba'` (fast) or `'numpy'` |
| `Model.calib` | Load `calib.txt` from parent of `image_dir` (KITTI format, MapAnything only) |
| `Loop.SIM3_Optimizer.lang_version` | `'cpp'` or `'python'` solver |

Dataset-specific configs inherit from base: `kitti.yaml`, `waymo.yaml`, `map_long_config.yaml`.

## Outputs (under `exps/<image_dir_path>/<timestamp>/`)

| File/Dir | Description |
|----------|-------------|
| `pcd/` | Per-chunk `.ply` point clouds + `combined_pcd.ply` |
| `camera_poses.txt` | C2W matrices (one 4×4 flattened per line) |
| `camera_poses.ply` | Camera pose positions visualized |
| `intrinsic.txt` | Per-frame `fx fy cx cy` (if available) |
| `sim3_opt_result.png` | Loop closure optimization trajectory plot |
| `loop_closures.txt` | Detected loop pairs |
| `base_config.yaml` | Copy of config used |

## Known Issues / Notes

- `faiss-gpu` install: replace with `faiss-gpu-cu11` or `faiss-gpu-cu12` if the bare package fails.
- `safetensors` version: downgrade to `0.5.3` if you see `torch has no attribute 'uint64'`.
- `libGL.so.1` error: `sudo apt-get install -y libgl1-mesa-glx`.
- Disk usage: ~50 GiB for 4500-frame sequences; ensure sufficient free space before running.
- `reference_frame_mid` has known bugs; keep disabled.
- For metric-scale models (MapAnything), set `using_sim3: False` (SE(3) alignment).
