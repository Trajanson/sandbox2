FROM python:3.6

RUN apt update
RUN apt upgrade -y

RUN pip3 install --upgrade pip

RUN apt-get install libjpeg-dev libtiff-dev libfreetype6-dev

RUN wget http://downloads.sourceforge.net/project/openjpeg.mirror/2.0.1/openjpeg-2.0.1.tar.gz
RUN tar xzvf openjpeg-2.0.1.tar.gz
WORKDIR openjpeg-2.0.1 
RUN apt-get install cmake -y
RUN cmake .
RUN make install
#install pillow
RUN pip3 install pillow

WORKDIR /

RUN pip3 install tensorflow keras

COPY . .






# FROM tensorflow/tensorflow:latest-py3


# RUN apt install build-essential checkinstall -y
# RUN apt install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev wget -y
# RUN wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tar.xz
# RUN tar xvf Python-3.6.0.tar.xz
# RUN cd Python-3.6.0/
# RUN ./configure
# RUN make altinstall



# RUN pip3 install --upgrade pip

# RUN pip3 install keras

# COPY . .