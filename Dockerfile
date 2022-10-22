FROM python:3.8-slim

ENV NVIDIA_VISIBLE_DEVICES all

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV CUDA_VERSION 11.0.0

ENV LD_LIBRARY_PATH="/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH"

RUN apt-get -y update

RUN apt -y install build-essential

RUN apt-get -y install git

# Install OpenCV requirements
RUN apt-get install ffmpeg libsm6 libxext6  -y

LABEL taost=taost

COPY . /AIR-Net/

RUN cd /AIR-Net/ && pip3 install -r requirements.txt

ENV CUDA_VISIBLE_DEVICES=all

ENV PORT=8811

EXPOSE 8811

# CMD jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --port $PORT
