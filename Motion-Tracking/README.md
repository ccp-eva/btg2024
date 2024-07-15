# Motion-Tracking
Friday, August 9, 2024, 11:30 – 15:00 by Tim-Joshua Andres & Arja Mentink

## Hands on
We are using different machine learning libraries to showcase how to build models for detecting/segmenting, tracking and identifying animals (in our example macaques), as well as some basic behavioural classification. While we are going to showcase how to train your own models, we will skip the actual training and provide you with the trained models - hence a GPU is not necessary for this session to work.

For the detection, segmentation and pose estimation we are going to use [Ultralytics YOLOv8](https://docs.ultralytics.com/) which provides a lot of resources in its documentation and is based on [Pytorch](https://pytorch.org/get-started/locally/). The identification network will be based on [Tensorflow](https://www.tensorflow.org/install).

We are going to work with the [MacaquePose-dataset](https://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html) (see also [10.3389/fnbeh.2020.581154](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2020.581154/full))

### Installation
#### Basic installation (CPU - only)
1. Start VSCode
2. Open a VSCode terminal
3. Create a conda environment called "motion_tracking" and install necessary packages
```
conda create --name motion_tracking python=3.9
conda activate motion_tracking
python -m pip install "tensorflow<2.11" ultralytics torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn
```
#### Installation with GPU (cuda needs to be supported)
If you are unsure whether your computer supports cuda you can check [this website](https://developer.nvidia.com/cuda-gpus) or just install the CPU supported packages.
##### Windows
1. Start VSCode
2. Open a VSCode terminal
3. Create a conda environment called "motion_tracking" and install necessary packages
```
conda create --name motion_tracking python=3.9
conda activate motion_tracking
conda install -c conda-forge cudatoolkit=11.1.1 cudnn=8.1.0.77
python -m pip install "tensorflow<2.11" ultralytics torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn
```

##### Ubuntu 22.04
1. Make sure [nvidia drivers are installed](https://ubuntu.com/server/docs/nvidia-drivers-installation)
2. Start VSCode
3. Open a VSCode terminal
4. Create a conda environment called "motion_tracking" and install necessary packages
```
conda create --name motion_tracking
conda activate motion_tracking
python3 -m pip install "tensorflow<2.11" ultralytics torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn
```
