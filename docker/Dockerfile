# FROM ubuntu:latest
FROM python:3.8.1
COPY models /exp/models
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install -r /exp/requirements.txt
WORKDIR /exp/models
CMD ["python3", "model2.py"]
