FROM python:3.8-slim

FROM nvidia/cuda:11.1.1-base-ubuntu18.04

FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ENV NVIDIA_VISIBLE_DEVICES all

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# ENV CUDA_VERSION 11.0.0

# ENV LD_LIBRARY_PATH="/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH"

RUN apt-get -y update

RUN apt -y install build-essential

RUN apt-get -y install git

<<<<<<< HEAD
LABEL taost=taost

COPY . /root/workspace/

RUN mkdir /root/workspace/out/

RUN cd /root/workspace/ && pip3 install -r requirements.txt
=======
# Install OpenCV requirements
# RUN apt-get install ffmpeg libsm6 libxext6  -y

LABEL taost=taost

COPY . /root/workspace/

RUN mkdir /root/workspace/out/
>>>>>>> d94e4a3ef58b8fc5a49ad305cbac55297d51fadc

# RUN cd /root/workspace/ && pip3 install -r requirements.txt

# RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

ENV CUDA_VISIBLE_DEVICES=all

ENV PORT=8811

EXPOSE 8811
