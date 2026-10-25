# Rabbit Streaming-Memory Trial Lab

All trial scripts, copied model code, configs, tests, manifests, and generated outputs belong here. Do not modify production training or dashboard modules for these experiments.

## Stage 0: Audit

Run from the repository root:

```powershell
python experiments/rabbit_streaming_memory/src/audit.py --full-hash --output experiments/rabbit_streaming_memory/manifests/audit_full.json
```

By default this scans the train and val subfolders under `dataset/rabbit/2026-10-22`, records media metadata, hashes both checkpoints, and strict-loads their weights into the copied RepViT-GRU definition. Pass `--data-root` to point to a different directory with `train/<class>` and `val/<class>` subfolders. `--full-hash` is required for a passing data audit: it hashes every video and checks byte-identical train/val leakage, which can take substantial disk I/O. The report is written even when videos are invalid, and the script exits nonzero so a data problem cannot silently pass. The initial audit and authorized dataset changes are retained in `manifests/`; the final audit `audit_final_2026-10-24.json` passes with 1040 train videos, 257 val videos, zero unreadable files, and zero cross-split duplicate groups.

## Local p95 benchmark

```powershell
.venv-vlm\Scripts\python.exe experiments/rabbit_streaming_memory/src/benchmark_model.py --device cuda --warmup 100 --iterations 1000
```

This measures batch-1 model-only latency for single-frame recurrent inference and independent 16-frame inference on the current local machine. For the five-repeat result, see `outputs/local_latency_verified_2026-10-24_5repeat_summary.json`; each round uses 100 warm-up and 1000 synchronized iterations. It uses synthetic tensors and is not an end-to-end video or deployment benchmark. The output records hardware/software details in `outputs/local_latency.json`. The default Anaconda Python on this host is CPU-only; use the project `.venv-vlm` for CUDA measurements.

## Provenance and isolation

`src/timm_gru_student.py` is the experiment-local copy of `models/timm_gru_student.py`; it adds stream helpers without changing parameter names. `audit.py` records source/copy hashes and the current Git HEAD/status. The lab writes only below this directory. Video files are read in place and are not copied.

## Stage 1: stream trainer and replay

The training entry point is isolated under `src/stream_trainer.py`. It requires a full-hash audit and an approved selection manifest. No source video is changed.

```powershell
.venv-vlm\Scripts\python.exe experiments/rabbit_streaming_memory/src/stream_trainer.py --mode t1 --device cuda --selection-manifest experiments/rabbit_streaming_memory/manifests/training_selection_approved.json --audit-manifest experiments/rabbit_streaming_memory/manifests/audit_final_2026-10-24.json
```

Modes are `t0` independent 16-frame windows with endpoint loss, `t1` 12-second causal blocks with uniform loss after a 5-second burn-in, and `t2` the same blocks with endpoint-only MiniROAD-style loss. The matrix launcher runs all three modes at seeds 42/43/44, writes one log per run, stops at the first failure, then creates a checkpoint/validation summary:

```powershell
.\experiments\rabbit_streaming_memory\run_training_matrix.ps1
```

Validation sampling is class-balanced up to 12 videos per class. The 2026-10-24 T0/T1/T2 x seeds 42/43/44 training matrix has completed; see `outputs/checkpoints/training_matrix_summary.csv`.

`validate_stream.py` checks data selection, contiguous decoding, checkpoint strict-load, stepwise/sequence equivalence, and loss shapes before training. All input videos are temporally sampled to the configured 8 FPS regardless of native 8/10/16/30 FPS. Short stream clips use their available frames and reduce burn-in to preserve one supervised step; T0 left-pads clips shorter than 16 frames. `replay_eval.py` consumes an isolated NPZ frame sequence and writes raw probabilities, reset reasons, and per-step latency.

The approved data selection and deletion record are in `manifests/training_selection_approved.json` and `manifests/dataset_changes_2026-10-24.json`. Selected checkpoints completed 30-video frozen evaluation. All three training candidates were below the original Oct20 baseline and failed the quality gate; do not promote them. An isolated 4 Hz dashboard-style replay now runs real checkpoints through the copied FusionAgent semantics and single-frame crop preprocessing on all 30 annotated videos. It still uses the training-repository YOLO and assumes one persistent track, so it is not end-to-end dashboard performance. Results and per-class metrics are in `outputs/dashboard_style_4hz/dashboard_style_comparison.csv` and `dashboard_style_by_class.csv`; the earlier 2 Hz fusion-only replay remains separately documented. See `outputs/trained_checkpoint_evaluation_2026-10-24.json` and `docs/rabbit_streaming_memory_experiment_plan.md`.


`src/dashboard_fusion_snapshot.py` is an isolated copy of Rabbit candidate/FSM semantics from the dashboard source. `src/apply_dashboard_fusion_snapshot.py` replays it over saved traces; this does not rerun dashboard detection, crops, or model inference and must not be treated as full online-system performance.
