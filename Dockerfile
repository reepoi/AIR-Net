FROM python:3.8-slim

ENV NVIDIA_VISIBLE_DEVICES all

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV CUDA_VERSION 11.0.0

ENV LD_LIBRARY_PATH="/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH"

RUN apt-get -y update

RUN apt -y install build-essential

RUN apt-get -y install git

RUN apt -y install wget

RUN wget https://github.com/neovim/neovim/releases/download/stable/nvim-linux64.deb

RUN apt install ./nvim-linux64.deb

RUN git clone --depth 1 https://github.com/wbthomason/packer.nvim ~/.local/share/nvim/site/pack/packer/start/packer.nvim

RUN git clone --depth 1 https://github.com/reepoi/.dotfiles.git ~/.config/.dotfiles

RUN apt-get -y install tmux

RUN apt-get -y install ripgrep

LABEL taost=taost

COPY . /root/workspace/

RUN mkdir /root/workspace/out/

RUN pip3 install virtualenvwrapper

RUN cd /root/workspace/ && pip3 install -r requirements.txt

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

ENV CUDA_VISIBLE_DEVICES=all

ENV PORT=8811

EXPOSE 8811

CMD bash
