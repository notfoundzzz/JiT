# JiT Local Minimal

This setup is for verifying that JiT can run locally on a single GPU with toy data.

## Environment

Create the minimal environment:

```bash
conda create -y -n jit-local python=3.10 pip
/home/zhahl/miniconda3/envs/jit-local/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
/home/zhahl/miniconda3/envs/jit-local/bin/pip install einops
```

## Run

```bash
cd /home/zhahl/JiT
./run_minimal_local.sh
```

The script:

- uses `toy_data/` instead of ImageNet
- runs on a single GPU
- uses `JiT-B/32` at `64x64`
- trains for `1` epoch with `batch_size=1`

Outputs are written to `output_local/`.

## Sample Images

Generate a few images from the latest checkpoint:

```bash
cd /home/zhahl/JiT
./run_minimal_sample.sh
```

Images are written to `samples_local/`.

## Visual Toy Experiment

This branch also includes a more structured toy experiment that is easier to inspect visually.

Train on synthetic geometric classes:

```bash
cd /home/zhahl/JiT
./run_visual_toy_train.sh
```

Sample images from the trained checkpoint:

```bash
cd /home/zhahl/JiT
./run_visual_toy_sample.sh
```

Outputs are written to `output_visual_toy/` and `samples_visual_toy/`.

If full sampling is still weak, inspect the denoising behavior directly:

```bash
cd /home/zhahl/JiT
./run_visual_toy_denoise.sh
```

Triptychs are written to `denoise_visual_toy/`.

## Single A100 ImageNet Run

For a single-GPU A100 sanity run on ImageNet:

```bash
cd /home/zhahl/JiT
export IMAGENET_PATH=/path/to/imagenet
./run_imagenet_a100.sh
```

This is a reduced real-data run, not the full 8-GPU training recipe from the original README.
Outputs are written to `output_imagenet_a100/`.
