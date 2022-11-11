conda run -n sd pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

cd src

git clone https://github.com/openai/CLIP.git
conda run -n sd pip install -e clip/.

git clone https://github.com/CompVis/taming-transformers.git
conda run -n sd pip install -e taming-transformers/.

git clone https://github.com/CompVis/latent-diffusion.git
conda run -n sd pip install -e latent-diffusion/.

wget -O /evaluate-images-to-feed-diffusion/model/waifu-diffusion-v1-4/kl-f8-anime2.ckpt \
    https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt
