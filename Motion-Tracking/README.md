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
### Outline of session
#### MacaquePose Dataset and YOLOv8 training
This part will have a look at the script *__convert_macpose_labels.py__* which converts the [MacaquePose dataset](https://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html) into [YOLO format](https://docs.ultralytics.com/datasets/segment/) and splits it into train, test and validation sets.
Further we will go through *__train_segmentation_model.py__* and *__train_poseestimation.py__* which showcase how to train a YOLO model on a dataset.
#### Segmenting and tracking animals in a video
In this part we run the code *__track_macaque_video.py__* which will segment (find the contour) and track (align semgentations from previous frames) animals in a video using the trained YOLO models and save the output. We use *__segmenting_track_images.py__* to extract cropped images from the video (using the segmentation mask) according to their track number to help us create a dataset for identification of individuals.
#### Training an identification network
We will walk through the steps of training your own identification network based on the EfficientNetv2 architecture and using a model pretrained on the [ChimpACT dataset](https://shirleymaxx.github.io/ChimpACT/) published by [Ma et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/57a95cd3898bf4912269848a01f53620-Paper-Datasets_and_Benchmarks.pdf) with the data extracted in the steps above. See *__identification_training.py__*
#### Pipeline segmenting, identifying and tracking macaques
This step combines all the previously created networks in a single pipeline. With *__pipeline_predict_video.py__* we can feed in the video to segment and track the macaques. Each segment is cropped and passed to the identification network to predict animal identity as well as the pose estiamation model to predict keypoints. These results are displayed and saved in two formats.
#### Social Network
Depending on the time we can also look at *__social_network.py__* to show you one approach to creating a social network from the data collected.
