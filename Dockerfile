FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

LABEL taost=taost

# Install OpenCV requirements
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV CUDA_VISIBLE_DEVICES=3

ENV PORT=8800

EXPOSE $PORT

WORKDIR .

CMD jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --port $PORT
