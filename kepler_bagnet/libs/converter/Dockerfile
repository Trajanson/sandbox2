FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install python3.6 libpython3.6 -y

RUN ln -s python3.6 /usr/bin/python3
RUN ln -s python3.6 /usr/bin/python

COPY . .