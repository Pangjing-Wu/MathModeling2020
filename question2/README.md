# Question 2

## Dependencies
- `pandas`
- `numpy`
- `torch==1.6.0`
- `torchvision`
- `tensorboard`
- `opencv`
- `PyAv`

## Usage
- `read_video.py`: generator dataset from original video
    - Put `Fog20200313000026.mp4` in this folder
    - Issue `python3 read_video.py`
    - Images sampled from video will be written in folder `./dataset`

- `train.py`: train ResNet18Rnn model
    - Training info from `tqdm` and `tensorboard`
