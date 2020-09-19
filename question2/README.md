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
    - Issue `python3 read_video.py --trans [normal|fft|color_map]`
    - Images sampled from video will be written in folder `./dataset_{trans}`

- `train.py`: train ResNet18Rnn model
    - Issue `python3 train.py --trans [normal|fft|color_map]` to train on different dataset
    - Training info from `tqdm` and `tensorboard`
