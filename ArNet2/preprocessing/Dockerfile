FROM python:3.6-buster
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -q update && \
apt-get install -q -y --no-install-recommends \
xorg \
unzip \
wget \
curl && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
# Install the MCR dependencies and some things we'll need and download the MCR
# from Mathworks -silently install it
RUN mkdir /mcr-install && \
mkdir /opt/mcr && \
cd /mcr-install && \
wget -q https://ssd.mathworks.com/supportfiles/downloads/R2020a/Release/4/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2020a_Update_4_glnxa64.zip && \
unzip -q MATLAB_Runtime_R2020a_Update_4_glnxa64.zip && \
rm -f MATLAB_Runtime_R2020a_Update_4_glnxa64.zip && \
./install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent && \
rm -rf /mcr-install

ENV LD_LIBRARY_PATH=/opt/mcr/v98/runtime/glnxa64
COPY . .
