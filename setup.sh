conda run -n sd pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
conda run -n sd pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

conda run -n sd pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

cd src
git clone https://github.com/CompVis/latent-diffusion.git

cd latent-diffusion
conda run -n sd pip install -e .

cd /evaluate-images-to-feed-diffusion
wget -O ./model/waifu-diffusion-v1-4/kl-f8-anime2.ckpt \
    https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt
