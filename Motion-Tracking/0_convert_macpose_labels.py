"""
Created on Wed May 10 15:51:34 2023

@author: J.Andres
"""

import os
import pandas as pd
import json
import numpy as np
import cv2
import itertools
import torch
import shutil
import itertools

draw_labeled_images = True  # will create a directory with labelled images from the annotations

df = pd.read_csv("./datasets/orig_macaquepose/annotations.csv")  # read in annotations file (MacaquePose)
path_images = "./datasets/orig_macaquepose/images/"  # path to image directory (MacaquePose)

path_out_labels = "./datasets/orig_macaquepose/labels/"  # path where to create directory with labels
path_out_images = "./datasets/orig_macaquepose/labeled_images/"  # path where to create directory with labelled images

split_dataset = True  # will create directories and yaml files in format for training and testing with yolo
path_2_dataset = "./datasets/macaque_pose/"  # path where to create dataset

# create directories or raise error if directories already exist
if not os.path.exists(path_out_labels):
    os.makedirs(path_out_labels)
    print("The new directory with subdirectories created: ", path_out_labels)
else:
    raise Exception("Directory already exists!")

if draw_labeled_images and not os.path.exists(path_out_images):
    os.makedirs(path_out_images)
    print("Will draw labeled images to: ", path_out_labels)
else:
    raise Exception("Directory already exists!")

# creating subdirectories for dataset or raise error if they already exist
if split_dataset:
    path_2_subdirs = ["pose/", "segmentation/"]
    paths_ttv = ["train/", "test/", "valid/"]

    if not os.path.exists(path_2_dataset):
        # Create a new directory because it does not exist
        os.makedirs(path_2_dataset)
        for f, b in itertools.product(path_2_subdirs, paths_ttv):
            os.makedirs(path_2_dataset + f + b)
            os.makedirs(path_2_dataset + f + b + "images/")
            os.makedirs(path_2_dataset + f + b + "labels/")
        print("The new directory with subdirectories created: ", path_2_dataset)
    else:
        raise Exception("Directory already exists!")


# define function - used later to ensure normalised values are between 0 and 1
def norm_01(val):
    return min(1., max(val, 0.))


# read in all possible keypoints
print(set([kd['name'] for r in df.keypoinbts for l in json.loads(r) for kd in l]))
all_keypoint_names = set([kd['name'] for r in df.keypoinbts for l in json.loads(r) for kd in l])
keypoint_num_dict = {n: i for i, n in enumerate(all_keypoint_names)}

# loop through annotations - each row is a new image
for i, r in df.iterrows():
    fname = r['image file name']  # file name of image
    img = cv2.imread(path_images + fname)
    ih, iw, ic = img.shape  # we need the image shape as yolo format consists of normalised coordinates

    ls_segmentations = []  # list holding segmentations in yolo format each element is an annotation
    ls_keypoints = []  # list holding keypoints in yolo format each element is an annotation
    for num_an in range(len(json.loads(r.segmentation))):
        seg = []
        for q in range(len(json.loads(r.segmentation)[num_an])):  # sometimes multiple segments are used to represent
            # the same element (e.g. when part is occluded by a tree) - here we connect these segments as we want a
            # single segment for each monkey

            ## Segments
            if q > 0:
                seg.append(seg[-1][:2])
            seg.append(np.array([[norm_01(crd[0] / iw), norm_01(crd[1] / ih)] for crd in
                                 json.loads(r.segmentation)[num_an][q]['segment']]).ravel().tolist())
            if q > 0:
                seg.append(seg[-1][:2])

        seg = list(itertools.chain(*seg))

        if draw_labeled_images:  # plot segmentation on image
            seg_pl = (np.array(seg).reshape(-1, 2) * np.array([iw, ih])).astype(np.int32)
            img = cv2.polylines(img, [seg_pl], True, [255, 255 * q, 0],
                                thickness=2)
        # add class at beginning of line - will only have one class
        seg.insert(0, 0)
        ls_segmentations.append(seg)

        ## Keypoints
        assert len(json.loads(r.keypoinbts)[num_an]) == 17
        keyp_pl = {kp['name']: kp['position'] for kp in json.loads(r.keypoinbts)[num_an]}
        if not all(value is None for value in keyp_pl.values()):
            keypts = [[0.0, 0.0, 0] if p is None else [norm_01(p[0] / iw), norm_01(p[1] / ih), 1] for p in
                      keyp_pl.values()]
            keypts = list(itertools.chain(*keypts))
            keypts_ar = np.array(keypts).reshape(17, 3)

            seg_ar = np.array(seg[1:]).reshape(int(len(seg[1:]) / 2), 2)
            wdth = np.max(seg_ar[:, 0]) - np.min(seg_ar[:, 0])
            cx = np.min(seg_ar[:, 0]) + wdth / 2
            hght = np.max(seg_ar[:, 1]) - np.min(seg_ar[:, 1])
            cy = np.min(seg_ar[:, 1]) + hght / 2

            if draw_labeled_images:
                img = cv2.rectangle(img, (int(iw * (cx - wdth / 2)), int(ih * (cy - hght / 2))),
                                    (int(iw * (cx + wdth / 2)), int(ih * (cy + hght / 2))),
                                    (255, 255, 0), thickness=5)
                for kpi in range(0, len(keypts) - 2, 3):
                    if keypts[kpi + 2] == 0:
                        continue
                    else:
                        img = cv2.circle(img, (int(keypts[kpi] * iw), int(keypts[kpi + 1] * ih)), 5, [255, 0, 255], 2)

            keypts[0:0] = [0, cx, cy, wdth, hght] # adds class and bounding box information
            ls_keypoints.append(keypts)

        if draw_labeled_images:
            cv2.imwrite(path_out_images + "lbl_" + fname, img)

    # writing labels to files
    if not os.path.exists(path_out_labels + 'labels_seg/'):
        # Create a new directory because it does not exist
        os.makedirs(path_out_labels + 'labels_seg/')
        print("The new directory with subdirectories created: ", path_out_labels + 'labels_seg/')
    with open(path_out_labels + 'labels_seg/' + fname[:-4] + '.txt', 'w+') as f:
        for l in ls_segmentations:
            f.write(" ".join(str(seg) for seg in l))
            f.write("\n")

    if not os.path.exists(path_out_labels + 'labels_kpt/'):
        # Create a new directory because it does not exist
        os.makedirs(path_out_labels + 'labels_kpt/')
        print("The new directory with subdirectories created: ", path_out_labels + 'labels_kpt/')
    with open(path_out_labels + 'labels_kpt/' + fname[:-4] + '.txt', 'w+') as f:
        for l in ls_keypoints:
            f.write(" ".join(str(kpt) for kpt in l))
            f.write("\n")

if split_dataset:

    # move/split/copy files to train, val, test - with 0.7,0.15,0.15 split
    full_dataset = [f[:-4] for f in os.listdir(path_images)]
    train_size = int(0.7 * len(full_dataset))
    test_size = int((len(full_dataset) - train_size) / 2)
    valid_size = len(full_dataset) - train_size - test_size
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(full_dataset,
                                                                               [train_size, test_size, valid_size])


    def copy_split(set, set_type_path):
        for fname in set:
            if os.path.exists(path_images + fname + ".jpg"):
                shutil.copy(path_images + fname + ".jpg",
                            path_2_dataset + "segmentation/" + set_type_path + "images/" + fname + ".jpg")
                shutil.copy(path_images + fname + ".jpg",
                            path_2_dataset + "pose/" + set_type_path + "images/" + fname + ".jpg")
            else:
                print(f"Image - {fname} not found!")

            if os.path.exists(path_out_labels + "labels_seg/" + fname + ".txt"):
                shutil.copy(path_out_labels + "labels_seg/" + fname + ".txt",
                            path_2_dataset + "segmentation/" + set_type_path + "labels/" + fname + ".txt")
            else:
                print(f"Segmentation label - {fname} not found!")

            if os.path.exists(path_out_labels + "labels_kpt/" + fname + ".txt"):
                shutil.copy(path_out_labels + "labels_kpt/" + fname + ".txt",
                            path_2_dataset + "pose/" + set_type_path + "labels/" + fname + ".txt")
            else:
                print(f"Pose label - {fname} not found!")

    copy_split(train_dataset, "train/")
    copy_split(test_dataset, "test/")
    copy_split(valid_dataset, "valid/")

    # automate config file
    conf_kpt_ls = ["#automatically created config file from converting macaquepose_v1\n",
                   "train: ./macaque_pose/pose/train/  # train images (relative to 'path')\n",
                   "val: ./macaque_pose/pose/valid/  # val images (relative to 'path')\n",
                   "test: ./macaque_pose/pose/test/  # test images (relative to 'path')\n",
                   "\n",
                   "# Keypoints\n",

                   f"kpt_shape: {[len(keypoint_num_dict), 3]}  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)]]\n",
                   "\n",
                   "# Classes\n",
                   "names:\n",
                   " 0: macaque\n"]

    with open("./costum_macaque_pose_kpt.yaml", 'w') as f:
        for l in conf_kpt_ls:
            f.write(l)

    conf_seg_ls = ["#automatically created config file from converting macaquepose_v1\n",
                   "train: ./macaque_pose/segmentation/train/ # train root directory\n",
                   "test: ./macaque_pose/segmentation/test/ #test root directory\n",
                   "val: ./macaque_pose/segmentation/valid/ #validation root directory\n",
                   "\n",
                   "# number of classes\n",
                   "nc: 1 # class names\n",
                   "\n",
                   "# class names\n",
                   "names: ['body']\n"]

    with open("./costum_macaque_pose_seg.yaml", 'w') as f:
        for l in conf_seg_ls:
            f.write(l)
