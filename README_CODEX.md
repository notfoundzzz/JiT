# README_CODEX

## Purpose

This file records the current JiT experiment state, what has already been tried, what worked, what failed, and how a future Codex should continue without repeating old work.

The main goal is:

- verify that the official pretrained JiT checkpoints work on their original generation task
- avoid wasting time redoing environment migration
- avoid treating JiT as a naive one-step black-box restorer unless explicitly testing that hypothesis


## Repository

- Repo path: `/home/zhahl/JiT-cloud-smoke`
- Branch: `cloud-v100-smoke`
- Current branch tip at time of writing: `d4f1913`

Recent relevant commits:

- `d4f1913` Add pretrained cloud generation helper scripts
- `7029076` Add pretrained JiT generation verification script
- `f4c69a7` Add local temporal and SR black-box test flows
- `0c6dcf8` Add pretrained JiT black-box local inference tools
- `63a2f19` Reduce repeated cloud training log noise
- `c437efa` Harden cloud V100 demo scripts


## Important Environment Facts

### Cloud

The cloud environment had already been validated before this note:

- Ubuntu 20.04 x86_64
- Tesla V100-PCIE-32GB
- driver `535.183.01`
- custom Python env on cloud:
  - `/data/Shenzhen/zhahongli/envs/jit-local/bin/python`
- validated package state on cloud:
  - `torch 2.5.1+cu118`
  - `torchvision 0.20.1+cu118`
  - `numpy 1.24.4`
  - `scipy 1.9.1`
  - `cv2 4.11.0`
  - `torch_fidelity` importable
  - `torch.cuda.is_available() == True`

Cloud CUDA pollution pitfall:

- before cloud runs, avoid the platform-provided `(PyTorch-2.0.0)` environment
- use:
  - `conda deactivate`
  - `unset LD_LIBRARY_PATH`
  - `unset CUDA_HOME`
  - `unset CUDA_PATH`

These cleanup steps were already baked into the cloud helper scripts.


### Local

The local machine is WSL2 on Windows with an RTX 3060 Laptop GPU.

Local JiT env:

- `/home/zhahl/miniconda3/envs/jit-local/bin/python`

Important note:

- one older WSL terminal/session showed broken CUDA state:
  - `cuInit 304`
  - `torch.cuda.is_available() == False`
  - `dmesg` contained `dxgkio_query_adapter_info: Ioctl failed: -22`
- later, the user opened a fresh Ubuntu terminal and reported:
  - `torch 2.5.1+cu118`
  - `cuda available: True`
  - `device count: 1`

Interpretation:

- the env itself is probably fine
- some older WSL sessions can become unhealthy
- future Codex should prefer a fresh terminal/session if local CUDA looks inconsistent

Also:

- `opencv-python 4.11.0.86` was installed into local `jit-local`
- `cv2` import should work in the healthy session


## Pretrained Checkpoints

Known local checkpoint directories not tracked by git:

- `/home/zhahl/JiT-cloud-smoke/JiT-l-32/`
- `/home/zhahl/JiT-cloud-smoke/JiT-H-32/`

The checkpoint that was explicitly tested:

- `/home/zhahl/JiT-cloud-smoke/JiT-l-32/checkpoint-last.pth`

Additional large local checkpoint:

- `/home/zhahl/JiT-cloud-smoke/JiT-H-32/checkpoint-last.pth`
- size observed locally: about `11G`

Operational warning for the H checkpoint:

- do not casually `torch.load(...)` it just to inspect keys or structure in a fragile local WSL session
- that kind of inspection already caused trouble once and is not needed for normal generation runs
- if H/32 is used, prefer directly running the intended generation script on cloud rather than doing extra local probing

Important structure fact about the official pretrained checkpoint:

- it contains:
  - `model`
  - `model_ema1`
  - `model_ema2`
- it does **not** contain `args`

This means:

- old scripts that rely on `checkpoint["args"]` do not work directly with the official pretrained checkpoint
- a Codex must manually construct the model configuration when loading the official checkpoint


## JiT Interface Summary

Core JiT forward interface:

- defined in [model_jit.py](/home/zhahl/JiT-cloud-smoke/model_jit.py)
- signature:
  - `model(x, t, y) -> output`

Semantics:

- `x`: image tensor `(N, C, H, W)`
- `t`: timestep/noise-strength tensor `(N,)`
- `y`: class label tensor `(N,)`
- `output`: image tensor `(N, C, H, W)`

The higher-level denoiser wrapper in [denoiser.py](/home/zhahl/JiT-cloud-smoke/denoiser.py):

- trains on a noisy mixture formulation
- generates by iterative sampling from noise

Important implication:

- JiT is not a plain supervised restorer
- directly using `model.net(z, t, y)` as a one-step restoration function is a strong distribution mismatch

Extra clarification:

- `y` is the class label condition, not a restoration observation
- in ImageNet-pretrained JiT, `y` means semantic category ids such as dog / cat / fish
- `y` is not a low-resolution frame, not a pair of video frames, and not any STVSR observation

About `label_drop_prob`:

- `label_drop_prob` is a training-side classifier-free guidance mechanism
- it is applied in `forward(...)` when `self.training == True`
- it is not the correct inference switch for "make the model unconditional"
- changing `label_drop_prob` at inference time does not convert normal generation into unconditional generation in the current scripts


## Official Pretrained Configuration Used

For the tested pretrained `JiT-L/32` checkpoint, the working assumed configuration was:

- `model=JiT-L/32`
- `img_size=512`
- `noise_scale=2.0`
- `cfg=2.5`
- `num_sampling_steps=50`
- EMA weights:
  - prefer `model_ema1`

This was based on the project README and confirmed to at least load and run.

For `JiT-H/32`, the README-recommended generation-side configuration is:

- `model=JiT-H/32`
- `img_size=512`
- `noise_scale=2.0`
- `cfg=2.2` or `2.3`
- `num_sampling_steps=50`
- EMA weights:
  - prefer `model_ema1`


## Official Evaluation Path From README

The upstream JiT README does provide an official pretrained evaluation path:

- `main_jit.py --evaluate_gen`

What it does:

- generates images from noise using `model.generate(labels)`
- saves images to an output folder
- evaluates generation quality with `torch_fidelity`
- reports:
  - FID
  - Inception Score

Relevant local implementation files:

- [main_jit.py](/home/zhahl/JiT-cloud-smoke/main_jit.py)
- [engine_jit.py](/home/zhahl/JiT-cloud-smoke/engine_jit.py)
- [prepare_ref.py](/home/zhahl/JiT-cloud-smoke/prepare_ref.py)

Important interpretation:

- the official evaluation path is for native conditional generation
- it does not validate one-step SR
- it does not validate one-step temporal interpolation


## Scripts Added for Pretrained Checkpoint Work

### 1. `verify_pretrained_jit.py`

Purpose:

- verify that the official pretrained checkpoint loads
- verify that one direct `model.net(z, t, y)` forward pass runs

This script does not perform full generation.


### 2. `generate_pretrained_samples.py`

Purpose:

- perform actual generation from noise using the official pretrained checkpoint
- this is the main script to verify that the pretrained model itself is healthy

Recommended use:

- run serially
- do not launch multiple large `JiT-L/32 @ 512, 50-step` jobs in parallel on a 6GB local GPU


### 3. `blackbox_jit_infer.py`

Purpose:

- test the hypothesis that JiT can be used as a one-step black-box image restoration module

Supported modes:

- `spatial`
- `temporal`

Interpretation:

- this is for experimentation only
- current results suggest this direct one-step usage is not a good path


### 6. `generate_pretrained_h32_samples.py`

Purpose:

- a JiT-H/32 counterpart to the pretrained generation helper
- defaulted for:
  - `JiT-H/32`
  - `img_size=512`
  - `noise_scale=2.0`
  - `cfg=2.3`

Important note:

- this script exists, but future Codex should be careful with local memory risk before running H/32 locally


### 7. `generate_pretrained_noise_triptych.py`

Purpose:

- visualize how different random noise seeds change JiT-L/32 generation
- save:
  - the displayed initial noise image
  - the generated image
  - a triptych image

Important note:

- this was prepared, but a local WSL CUDA failure blocked execution in the Codex-side session


### 8. `run_cloud_pretrained_generate.sh`

Purpose:

- cloud-side serial generation helper for pretrained checkpoints
- designed to avoid training and avoid black-box restoration distractions

Behavior:

- checks `nvidia-smi`
- checks Python/CUDA availability
- serially runs `generate_pretrained_samples.py` over a grid of `label x seed`
- writes logs to `./logs`

Default focus:

- `JiT-L/32`
- can be redirected to `JiT-H/32` by overriding:
  - `MODEL_NAME`
  - `CFG_SCALE`
  - `CHECKPOINT_PATH`


### 4. `download_test_data.py`

Purpose:

- download a few Kodak images for local testing


### 5. `download_video_test_data.py`

Purpose:

- download a small video sample
- extract a pair of frames for temporal interface testing


## Cloud Scripts Already Adjusted

These cloud scripts were already hardened:

- [run_cloud_v100_smoke.sh](/home/zhahl/JiT-cloud-smoke/run_cloud_v100_smoke.sh)
- [run_cloud_v100_smoke_sample.sh](/home/zhahl/JiT-cloud-smoke/run_cloud_v100_smoke_sample.sh)
- [run_cloud_v100_visual_toy.sh](/home/zhahl/JiT-cloud-smoke/run_cloud_v100_visual_toy.sh)
- [run_cloud_v100_display_demo.sh](/home/zhahl/JiT-cloud-smoke/run_cloud_v100_display_demo.sh)

Important behaviors already added:

- direct use of `JIT_PYTHON=/data/Shenzhen/zhahongli/envs/jit-local/bin/python`
- log files written to `./logs`
- terminal output kept short
- cleanup of:
  - `LD_LIBRARY_PATH`
  - `CUDA_HOME`
  - `CUDA_PATH`
- display demo can reuse an existing checkpoint instead of retraining every run


## What Was Successfully Completed

### A. Smoke and display-first cloud path

Already known from earlier work:

- toy smoke training can complete
- checkpoint can be produced
- display-first demo path was switched to `JiT-B/16` where needed


### B. Official pretrained checkpoint can load

Confirmed:

- the pretrained `JiT-L/32` checkpoint loads if the model config is manually specified
- a direct one-step `model.net(z, t, y)` forward pass runs


### C. Official pretrained checkpoint can generate a plausible image

This is the most important validation already completed.

Generated image:

- [label_000_seed_0_00.png](/home/zhahl/JiT-cloud-smoke/generated_pretrained/label0_seed0/label_000_seed_0_00.png)

Interpretation:

- the official pretrained checkpoint itself is not broken
- JiT is able to produce a plausible ImageNet-like sample on its native task


### D. Label sweep and seed sweep behaved as expected

Additional serial local generation checks were completed for `JiT-L/32`.

Label sweep, fixed `seed=0`:

- [label_000_seed_0_00.png](/home/zhahl/JiT-cloud-smoke/generated_pretrained/label_sweep/label_0_seed0/label_000_seed_0_00.png)
- [label_207_seed_0_00.png](/home/zhahl/JiT-cloud-smoke/generated_pretrained/label_sweep/label_207_seed0/label_207_seed_0_00.png)
- [label_281_seed_0_00.png](/home/zhahl/JiT-cloud-smoke/generated_pretrained/label_sweep/label_281_seed0/label_281_seed_0_00.png)

Observed interpretation:

- different labels produce clearly different ImageNet-like semantics
- the tested outputs were consistent with distinct semantic categories

Seed sweep, fixed `label=0`:

- [label_000_seed_0_00.png](/home/zhahl/JiT-cloud-smoke/generated_pretrained/seed_sweep/label_0_seed_0/label_000_seed_0_00.png)
- [label_000_seed_1_00.png](/home/zhahl/JiT-cloud-smoke/generated_pretrained/seed_sweep/label_0_seed_1/label_000_seed_1_00.png)
- [label_000_seed_2_00.png](/home/zhahl/JiT-cloud-smoke/generated_pretrained/seed_sweep/label_0_seed_2/label_000_seed_2_00.png)

Observed interpretation:

- with label fixed, changing the seed changes within-class instance details rather than destroying the semantic class
- this further supports that the pretrained generation path is functioning normally


## What Was Tried and Failed

### A. One-step spatial black-box restoration

Test artifacts:

- [kodim03_input.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/kodak_spatial/kodim03_input.png)
- [kodim03_blackbox_input.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/kodak_spatial/kodim03_blackbox_input.png)
- [kodim03_pred.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/kodak_spatial/kodim03_pred.png)

And the x4 degradation version:

- [kodim12_x4_input.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/kodak_sr_x4/kodim12_x4_input.png)
- [kodim12_x4_blackbox_input.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/kodak_sr_x4/kodim12_x4_blackbox_input.png)
- [kodim12_x4_pred.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/kodak_sr_x4/kodim12_x4_pred.png)

Observed result:

- outputs are blurry, distorted, and strongly biased by the generative prior
- they are not faithful restorations


### B. One-step temporal black-box interpolation

Test artifacts:

- [frame_00040_to_frame_00042_frame0.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/video_temporal/frame_00040_to_frame_00042_frame0.png)
- [frame_00040_to_frame_00042_frame1.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/video_temporal/frame_00040_to_frame_00042_frame1.png)
- [frame_00040_to_frame_00042_mean_input.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/video_temporal/frame_00040_to_frame_00042_mean_input.png)
- [frame_00040_to_frame_00042_pred.png](/home/zhahl/JiT-cloud-smoke/blackbox_outputs/video_temporal/frame_00040_to_frame_00042_pred.png)

Observed result:

- not a meaningful interpolated middle frame
- output behaves more like a distorted generative rewrite than frame interpolation


## Current Interpretation

This is the key conclusion a future Codex should keep in mind:

- the pretrained JiT model appears to work on its intended generation task
- the direct one-step black-box restoration idea appears **not** to work well

This strongly suggests:

- the poor restoration results are caused by task/distribution mismatch
- not by a broken checkpoint
- not by a broken JiT architecture

More specifically:

- JiT was trained as a class-conditional generative / denoising model
- it was not trained as a faithful one-step super-resolution model
- it was not trained as a faithful one-step frame interpolation model
- feeding bicubic-upsampled LR images or mean-of-two-frames inputs into `model.net(z, t, y)` is not close to the training distribution

This also means:

- if JiT is ever used for STVSR, it should be treated as a generative prior inside an iterative constrained procedure
- it should not be treated as a one-call black-box predictor `output = net(input, t, y)`


## Recommended Next Steps

### If the goal is to understand pretrained JiT behavior

Do this next:

- use [generate_pretrained_samples.py](/home/zhahl/JiT-cloud-smoke/generate_pretrained_samples.py)
- run multiple `label / seed` combinations
- run them serially, not in parallel
- build intuition:
  - `label` controls semantic class
  - `seed` controls within-class variation
- if cloud resources are available, prefer the cloud helper script for longer serial sweeps


### If the goal is to do restoration / inverse problems with JiT

Do **not** keep pushing the one-step black-box path as the main route.

A more defensible next step would be:

- iterative sampling with observation constraints
- inverse-problem style restoration
- not direct single forward `model.net(z, t, y)`


### If the goal is practical SR / interpolation results soon

Use a task-matched baseline first:

- SR:
  - ESRGAN
  - SwinIR
- interpolation:
  - RIFE
  - FILM
  - IFRNet

Then consider whether JiT should act as a prior or refinement module.


## Operational Advice for Future Codex

- do not redo conda-pack or environment migration unless the environment is actually broken
- prefer the fresh local terminal/session if CUDA seems inconsistent
- if local WSL CUDA becomes unstable again, consider using cloud V100 for formal generation experiments
- avoid launching multiple heavy `JiT-L/32` generation jobs in parallel on the local 6GB GPU
- do not inspect the `11G` H/32 checkpoint structure locally unless there is a strong reason
- do not perform high-risk local memory operations without warning the user first
- keep terminal output short and write detailed logs to files when working on cloud scripts

WSL-specific reminder:

- when local CUDA falls into the `Error 304` / `dxgkio_query_adapter_info: Ioctl failed: -22` state, the effective fix is usually a fresh healthy WSL session
- often this means:
  - close old terminals
  - `wsl --shutdown`
  - reopen Ubuntu
- if Codex's own execution environment still sees the broken state, do not keep pushing local runs; switch focus to cloud or ask the user to run the command in their healthy terminal


## Current Git State

At the time of this update:

- current branch: `cloud-v100-smoke`
- local tip includes:
  - `d4f1913` Add pretrained cloud generation helper scripts

Important push note:

- the commit exists locally
- a Codex-attempted `git push` failed because the current execution environment could not reach GitHub
- first failure mode:
  - bad proxy path to `192.168.32.1:7897`
- second failure mode:
  - after removing proxy variables for that one command only, DNS resolution for `github.com` still failed

Important safety clarification:

- Codex did not permanently remove the user's proxy settings
- proxy variables were only unset temporarily for a single command invocation via `env -u ... git push ...`


## Useful Commands

### Verify local CUDA in the healthy session

```bash
/home/zhahl/miniconda3/envs/jit-local/bin/python - <<'PY'
import torch
print(torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```


### Generate one pretrained sample locally

```bash
cd /home/zhahl/JiT-cloud-smoke && /home/zhahl/miniconda3/envs/jit-local/bin/python generate_pretrained_samples.py --checkpoint /home/zhahl/JiT-cloud-smoke/JiT-l-32/checkpoint-last.pth --output_dir /home/zhahl/JiT-cloud-smoke/generated_pretrained/label0_seed0 --model JiT-L/32 --img_size 512 --noise_scale 2.0 --cfg 2.5 --num_sampling_steps 50 --device cuda --labels 0 --seed 0
```


### Cloud display demo

```bash
conda deactivate >/dev/null 2>&1 || true; cd /data/Shenzhen/zhahongli/JiT-cloud-v100-smoke && git fetch origin cloud-v100-smoke && git checkout cloud-v100-smoke && git pull --ff-only origin cloud-v100-smoke && DISPLAY_FORCE_TRAIN=0 bash run_cloud_v100_display_demo.sh
```


## Final Summary

The single most important state summary is:

- pretrained JiT generation appears valid
- direct one-step black-box restoration appears invalid
- the environment work should not be restarted from scratch
- future effort should focus on either:
  - better pretrained generation verification, or
  - a more principled inverse-problem formulation


## 2026-03-29 Addendum: Qwen-VL Restoration Branch

This section records the current state of the newer image-to-image restoration effort so the next Codex instance can continue without reconstructing context.

### Current Branches

- main earlier restoration branch:
  - `JiT-image-to-image`
- newer experimental branch with Qwen2-VL visual encoder:
  - `qwen-vl-restoration-encoder`

Relevant pushed commits on `qwen-vl-restoration-encoder`:

- `d17fc0c` Add Qwen2-VL restoration encoder prototype
- `9be2afd` Fix Qwen2-VL projector dtype
- `1394868` Improve paired data generation progress logging
- `0dcac68` Force unbuffered paired data logging
- `636485a` Sample source images during paired data scan


### What Was Changed on `qwen-vl-restoration-encoder`

The lightweight CNN condition encoder was replaced with a Qwen2-VL visual branch:

- new file:
  - [condition_encoder_qwenvl.py](/home/zhahl/JiT-cloud-smoke/condition_encoder_qwenvl.py)
- it uses:
  - `Qwen2VLModel.from_pretrained(model_path).visual`
  - `AutoProcessor.from_pretrained(model_path).image_processor`
- the visual branch outputs are projected to JiT hidden size through a learned projector

Current intended restoration path:

- `LQ image`
- `Qwen2-VL image processor`
- `Qwen2-VL visual encoder`
- `projector`
- `cond_tokens / cond_global`
- `JiT restoration backbone`

Other important changes:

- JiT backbone and Qwen visual backbone are frozen
- only projector / bridge / norm style layers are trainable
- restoration loss was simplified to primarily:
  - `L1(x_pred, HQ)`
- training and inference scripts now expect:
  - `QWEN_MODEL_PATH`

Files touched for this branch:

- [condition_encoder_qwenvl.py](/home/zhahl/JiT-cloud-smoke/condition_encoder_qwenvl.py)
- [model_jit_restoration.py](/home/zhahl/JiT-cloud-smoke/model_jit_restoration.py)
- [denoiser_restoration.py](/home/zhahl/JiT-cloud-smoke/denoiser_restoration.py)
- [main_jit_restoration.py](/home/zhahl/JiT-cloud-smoke/main_jit_restoration.py)
- [infer_jit_restoration.py](/home/zhahl/JiT-cloud-smoke/infer_jit_restoration.py)
- [run_train_restoration_h32.sh](/home/zhahl/JiT-cloud-smoke/run_train_restoration_h32.sh)
- [run_infer_restoration_triptych.sh](/home/zhahl/JiT-cloud-smoke/run_infer_restoration_triptych.sh)


### Qwen2-VL Model Location

On local machine:

- model directory:
  - `/home/zhahl/models/Qwen2-VL-2B-Instruct`

This local model was downloaded successfully and used to inspect structure.

Important verified structure facts:

- top-level model:
  - `Qwen2VLForConditionalGeneration`
- core module:
  - `model`
- inside `model`:
  - `visual`
  - `language_model`
- the visual branch used here is:
  - `m.model.visual`
- visual forward signature:
  - `forward(hidden_states, grid_thw, **kwargs)`
- visual config showed:
  - vision hidden size `1536`
- JiT-H/32 hidden size is `1280`
- therefore a projector from `1536 -> 1280` is required

The next Codex should not waste time rediscovering that the Qwen visual branch location is:

- `Qwen2VLModel(...).visual`


### Local Environment Additions for Qwen Branch

The local `jit-local` environment was extended with these packages:

- `transformers==5.4.0`
- `accelerate==1.13.0`
- `tokenizers==0.22.2`
- `sentencepiece==0.2.1`
- `regex==2026.2.28`
- `psutil==7.2.2`
- `huggingface_hub==1.6.0`
- `safetensors==0.7.0`

A tarball was prepared locally for cloud-side environment patching:

- [qwen_jit_deps.tar.gz](/home/zhahl/qwen_jit_deps.tar.gz)

It contains the package directories and matching `.dist-info` entries for the packages above.

The intended cloud-side extraction target is:

- `/data/Shenzhen/zhahongli/envs/jit-local/lib/python3.10/site-packages`

Suggested cloud extraction command:

```bash
cd /data/Shenzhen/zhahongli/envs/jit-local/lib/python3.10/site-packages && tar -xzf /data/Shenzhen/zhahongli/qwen_jit_deps.tar.gz
```

Suggested cloud verification command:

```bash
/data/Shenzhen/zhahongli/envs/jit-local/bin/python -c "import transformers, accelerate, tokenizers, sentencepiece, huggingface_hub, safetensors; print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('tokenizers', tokenizers.__version__); print('sentencepiece', sentencepiece.__version__); print('huggingface_hub', huggingface_hub.__version__); print('safetensors', safetensors.__version__)"
```


### Current Cloud Situation for Qwen Branch

Latest known blocker on cloud Qwen smoke:

- code and model directory were present
- failure became:
  - `ModuleNotFoundError: No module named 'transformers'`

This means:

- the code sync was fine
- the Qwen model upload was fine
- the cloud Python environment still lacked the new packages

So the next Codex should first ensure the cloud environment is patched before debugging Qwen model code further.


### Local Validation Status of Qwen Branch

What was already validated locally:

- importing and loading `Qwen2-VL-2B-Instruct` succeeded
- locating the visual branch succeeded
- constructing `RestorationDenoiser` with Qwen encoder succeeded
- trainable parameter count after freezing printed successfully

Useful successful local command:

```bash
cd /home/zhahl/JiT-cloud-smoke && /home/zhahl/miniconda3/envs/jit-local/bin/python -c "from types import SimpleNamespace; from denoiser_restoration import RestorationDenoiser; args=SimpleNamespace(model='JiT-B/32', img_size=256, attn_dropout=0.0, proj_dropout=0.0, qwen_model_path='/home/zhahl/models/Qwen2-VL-2B-Instruct', P_mean=-0.8, P_std=0.8, t_eps=0.05, noise_scale=1.0, sampling_method='heun', num_sampling_steps=50, recon_weight=1.0, ema_decay1=0.9999, ema_decay2=0.9996); m=RestorationDenoiser(args); print('trainable_params', sum(p.numel() for p in m.parameters() if p.requires_grad)); print('total_params', sum(p.numel() for p in m.parameters()))"
```

Observed output summary:

- trainable params: about `1.78M`
- total params: about `800M`

What failed locally:

- full forward validation did not complete in Codex because local JiT initialization still hits a pre-existing device hardcode in:
  - [util/model_util.py](/home/zhahl/JiT-cloud-smoke/util/model_util.py)
- the problematic pattern is `.cuda()` inside JiT utility setup
- this is a separate JiT device-hardcode issue, not a Qwen branch logic issue


### Paired Data Script Improvements

The paired-data generation scripts were made more suitable for large source directories:

- [create_paired_restoration_data.py](/home/zhahl/JiT-cloud-smoke/create_paired_restoration_data.py)
- [run_prepare_restoration_data.sh](/home/zhahl/JiT-cloud-smoke/run_prepare_restoration_data.sh)

Important improvements:

- Python is now launched with `-u` for unbuffered logs
- scan progress is printed during source directory traversal
- save progress prints:
  - saved pairs
  - rate
  - ETA
  - skipped count
- source image enumeration no longer stores the entire very large source set
- instead, it now performs sampling during traversal

New useful parameters:

- `SCAN_LOG_FREQ`
- `SAVE_LOG_FREQ`
- `MAX_SOURCE_IMAGES`

Current behavior:

- the script traverses the source tree
- performs reservoir-style sampling of source images
- keeps at most `MAX_SOURCE_IMAGES` candidates
- then generates paired data from that sampled subset

Recommended large-run example:

```bash
MAX_SOURCE_IMAGES=5000 SCAN_LOG_FREQ=2000 SAVE_LOG_FREQ=200 NUM_SAMPLES=20000 SOURCE_DIR=/your/source_dir OUTPUT_DIR=/your/output_dir bash run_prepare_restoration_data.sh
```

Important reminder:

- running with the same `OUTPUT_DIR` will overwrite existing `sample_XXXXX.png` files
- use a fresh output directory to preserve prior paired sets


### Restoration DDP / Multi-GPU Status

The multi-GPU training helper on the older restoration branch was improved so that:

- multi-GPU launch uses:
  - `python -m torch.distributed.run`
- not the broken direct `torchrun` binary path

Multi-GPU status summary:

- DDP / NCCL startup was verified to work
- ranks `0..7` were observed on cloud
- one DDP issue about unused parameters was fixed by enabling:
  - `find_unused_parameters=True`

However, a later multi-GPU run still failed at checkpoint save time:

- failure was not training forward/backward itself
- it was GPU OOM during:
  - `copy.deepcopy(model_without_ddp.state_dict())`
- location:
  - [util/misc.py](/home/zhahl/JiT-cloud-smoke/util/misc.py)

Conclusion:

- multi-GPU infrastructure is mostly working
- current remaining issue on that path is checkpoint-saving memory overhead
- the next Codex should change save logic to CPU-side checkpoint assembly instead of GPU-side deepcopy


### Recommended Next Steps for Future Codex

For the Qwen branch, do these in order:

1. patch the cloud `jit-local` environment with the packaged dependencies
2. verify cloud imports of `transformers` and `Qwen2VLModel`
3. run a single-GPU Qwen smoke
4. only after that, debug any remaining shape/device issues
5. do not jump to multi-GPU Qwen training before single-GPU smoke is clean

Suggested cloud smoke after env patch:

```bash
cd /data/Shenzhen/zhahongli/JiT-image-to-image && git fetch origin qwen-vl-restoration-encoder && git checkout qwen-vl-restoration-encoder && git pull --ff-only origin qwen-vl-restoration-encoder
```

```bash
QWEN_MODEL_PATH=/data/Shenzhen/zhahongli/models/Qwen2-VL-2B-Instruct DATA_PATH=/data/Shenzhen/zhahongli/JiT-image-to-image/paired_smoke_qwen OUTPUT_DIR=/data/Shenzhen/zhahongli/JiT-image-to-image/output_smoke_qwen EPOCHS=1 NUM_GPUS=1 BATCH_SIZE=1 bash run_train_restoration_h32.sh
```

If that smoke fails, the next Codex should first inspect:

- Qwen model loading
- processor/image pre-processing behavior on cloud
- JiT device hardcodes
- and only later larger-scale training behavior
