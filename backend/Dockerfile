FROM python:3.10
RUN apt-get update \
    && apt-get install tesseract-ocr -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
ADD . /app

RUN mkdir -p /app/api/models/
RUN curl -o /app/api/models/Resnet152_ep_6_from_10.pth https://huggingface.co/Nikrir/Resnet152_ep_6_from_10.pth/resolve/main/Resnet152_ep_6_from_10.pth

RUN pip install -r requirements.txt