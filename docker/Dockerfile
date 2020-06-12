FROM tensorflow/tensorflow:1.14.0-gpu-py3

ADD https://raw.githubusercontent.com/senguptaumd/Background-Matting/master/requirements.txt /requirements_bgm.txt
ARG DEBIAN_FRONTEND=noninteractive
# Background-Matting
RUN apt-get update && apt-get install python-opencv -y && pip install -r /requirements_bgm.txt torch==1.4.0 torchvision==0.5.0
RUN apt-get install git -y && git clone https://github.com/tensorflow/models.git /tf-models 
ENV PYTHONPATH "${PYTHONPATH}:/tf-models/research:/tf-models/research/slim"
# Pose2Seg
RUN git clone https://github.com/liruilong940607/Pose2Seg.git /Pose2Seg
RUN sed -i '/torchvision-nightly/d' /Pose2Seg/requirements.txt && sed -i '/pkg-resources/d' /Pose2Seg/requirements.txt && sed -i '/torch-nightly/d' /Pose2Seg/requirements.txt && pip install --upgrade cython && pip install -r /Pose2Seg/requirements.txt
COPY infer.py /Pose2Seg/infer.py

