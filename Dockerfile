FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install essentials
RUN apt-get update && apt-get install -y curl git wget unzip python-pip libgl1-mesa-dev

# Setup conda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -o Miniconda3-py39_4.10.3-Linux-x86_64.sh
RUN bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b
ENV PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN conda init bash && conda update -y conda

RUN git clone https://github.com/discus0434/evaluate-images-to-feed-diffusion.git
WORKDIR /evaluate-images-to-feed-diffusion

RUN conda env create -f environment.yaml && echo "source activate sd" > ~/.bashrc
RUN chmod +x ./setup.sh && ./setup.sh
