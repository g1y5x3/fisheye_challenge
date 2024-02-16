FROM --platform=linux/amd64 ultralytics/ultralytics:latest

RUN apt-get update && \
    apt-get install -y ssh && \ 
    apt-get install -y vim  && \
    apt-get install -y git  && \
    apt-get -y install python3-pip && \
    pip install wandb

COPY datasets/Fisheye8K_all_including_train.tar.gz /workspace/FishEye8k/