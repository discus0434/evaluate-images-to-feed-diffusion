cd src
git clone https://github.com/CompVis/latent-diffusion.git

cd latent-diffusion
conda run -n sd pip install -e .

cd /evaluate-images-to-feed-diffusion
wget -O ./model/waifu-diffusion-v1-4/kl-f8-anime2.ckpt https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt
