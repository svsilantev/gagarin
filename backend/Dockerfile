FROM python:3.10

ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt

#RUN curl -O https://huggingface.co/Nikrir/Resnet152_ep_6_from_10.pth/resolve/main/Resnet152_ep_6_from_10.pth?download=true
#RUN mkdir /app/api/models/
#RUN mv Resnet152_ep_6_from_10.pth /app/api/model/