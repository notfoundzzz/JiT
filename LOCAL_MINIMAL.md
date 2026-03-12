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
