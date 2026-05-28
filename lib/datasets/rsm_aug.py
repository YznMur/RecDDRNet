# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
from PIL import Image
import random
import torch
from torch.nn import functional as F
from .base_dataset import BaseDataset


class RSM(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=4,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(RSM, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate,
                                  scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(root+list_path)]
        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {
            -1: ignore_label, 0: 0, 1: 1, 2: 2, 3: 3,
            4: ignore_label, 5: ignore_label, 6: ignore_label,
            7: ignore_label, 8: ignore_label, 9: ignore_label,
            10: ignore_label, 11: ignore_label, 12: ignore_label,
            13: ignore_label, 14: ignore_label, 15: ignore_label,
            16: ignore_label
        }
        self.class_weights = None

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({"img": image_path[0], "name": name})
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label


    # --------------------------------------------------
    # Improved Photometric Augmentations
    # --------------------------------------------------

    def random_color_jitter(self, image):
        if random.random() < 0.8:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

            hsv[:, :, 0] += random.uniform(-10, 10)
            hsv[:, :, 1] *= random.uniform(0.8, 1.2)
            hsv[:, :, 2] *= random.uniform(0.8, 1.2)

            hsv = np.clip(hsv, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image

    def random_gamma(self, image):
        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.3)
            inv = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv) * 255
                for i in np.arange(256)
            ]).astype("uint8")
            image = cv2.LUT(image, table)
        return image

    def random_noise(self, image):
        if random.random() < 0.5:
            noise = np.random.normal(0, 8, image.shape)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def random_blur(self, image):
        if random.random() < 0.85:
            return image
        return cv2.GaussianBlur(image, (3, 3), 0)

    def random_rotation(self, image, label):
        if random.random() < 0.7:
            return image, label

        angle = random.uniform(-5, 5)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        image = cv2.warpAffine(
            image, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        label = cv2.warpAffine(
            label, rot_mat, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.ignore_label
        )

        return image, label


    # --------------------------------------------------
    # GET ITEM
    # --------------------------------------------------

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(
            os.path.join(self.root, 'rsm', item["img"]),
            cv2.IMREAD_COLOR
        )
        size = image.shape

        if 'test' in self.list_path:
            image = cv2.resize(image, (2048, 1024))
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name

        label = cv2.imread(
            os.path.join(self.root, 'rsm', item["label"]),
            cv2.IMREAD_GRAYSCALE
        )
        label = self.convert_label(label)

        # ---------------- Multi-scale ----------------
        if self.multi_scale:
            image, label = self.multi_scale_aug(
                image, label,
                rand_crop=True
            )


        # ---------------- Geometry ----------------
        image, label = self.random_rotation(image, label)

        if self.flip and random.random() < 0.5:
            image = image[:, ::-1, :]
            label = label[:, ::-1]

        # ---------------- Photometric ----------------
        image = self.random_color_jitter(image)
        image = self.random_gamma(image)
        image = self.random_noise(image)
        image = self.random_blur(image)

        # ---------------- Normalize ----------------
        image = self.input_transform(image)
        label = self.label_transform(label)
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name
