# Human background subtraction
This repository contains developed method for static background subtraction from videos with people. 

The pipeline consists of a few steps: keypoints extraction using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), segmentation using [Pose2Seg](https://github.com/liruilong940607/Pose2Seg), background calculation using script `extract_bg.py`

The core is a modified version of [Background-Matting](https://github.com/senguptaumd/Background-Matting) approach.

## Docker

First, install `nvidia-docker` using instructions from [the repo](https://github.com/NVIDIA/nvidia-docker)

To build docker container, run

```python
cd docker
docker build -t human-bgs .
```

And run using the following command, where `CODE_DIR` is local directory of this repository, `DATA_DIR` is a folder containing a video required for processing and `VID_NAME` is a name of the video. Then, the output archive with videos can be found in `DATA_DIR/result.zip`

```python
docker run --gpus all -v $CODE_DIR:/human-bgs $DATA_DIR:/data -it human-bgs bash -c "cd /human-bgs && bash main.sh /data/$VID_NAME`
```
