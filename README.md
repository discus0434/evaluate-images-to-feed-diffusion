# Evaluate Images to Feed Diffusion

## Simple Description

A small notebook to:

- Preprocess images
- Evaluate how difficult the model grasp the features of your images
    - evaluated mainly by perceptual and GAN's discriminator loss.

## Installation

### Docker

1. Just build `Dockerfile`.

```bash
docker build . -t eval_images -t eval_images
```

2. Then, run.

```bash
docker run -it --runtime=nvidia --gpus all -d --restart=always eval_images:latest bash
```

### Conda

1. Create virtual environment from `environment.yaml`.

```bash
conda env create -f=environment.yaml
conda activate sd
```

2. Run `setup.sh`.

```bash
chmod +x ./setup.sh
./setup.sh
```
