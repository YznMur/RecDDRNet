# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified to support sequence-based training
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

class RSM(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=5,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 sequence=False,
                 sequence_len=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(RSM, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        self.sequence = sequence
        self.sequence_len = max(1, sequence_len)
        
        file_path = os.path.join(root, list_path)
        self.img_list = [line.strip().split() for line in open(file_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {-1: ignore_label, 0: 0, 
                              1: 1, 2: 2, 
                              3: 3, 4: 4}
    
        # self.label_mapping = {i: i for i in range(17)}
        self.class_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 
                                        3.0]).cuda()
        # self.class_weights = None
        
    def read_files(self):
        files = []
        if 'test' in self.list_path or 'camera' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                # Handle both single frames and sequences
                # Single frame: [img_path, label_path]
                # Sequence: [img1, label1, img2, label2, img3, label3, img4, label4]
                if len(item) == 2:
                    # Single frame format
                    image_path, label_path = item
                    name = os.path.splitext(os.path.basename(label_path))[0]
                    files.append({
                        "img": image_path,
                        "label": label_path,
                        "name": name,
                        "weight": 1,
                        "is_sequence": False
                    })
                elif len(item) % 2 == 0 and len(item) > 2:
                    # Sequence format: img1 label1 img2 label2 ...
                    # Extract sequence of frames
                    sequence = []
                    for i in range(0, len(item), 2):
                        sequence.append({
                            "img": item[i],
                            "label": item[i+1]
                        })
                
                    # Use the label of the last frame as the name
                    name = os.path.splitext(os.path.basename(sequence[-1]["label"]))[0]
                
                    files.append({
                        "img": sequence,  # Store entire sequence
                        "label": sequence,
                        "name": name,
                        "is_sequence": True,
                        "weight": 1
                    })
                else:
                    # Invalid format, skip
                    print(f"Warning: Invalid item format with {len(item)} elements, skipping")
                    continue
        return files
        
    def __len__(self):
        if self.sequence and self.sequence_len > 1:
            return max(0, len(self.files) - self.sequence_len + 1)
        return len(self.files)

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        if self.sequence and self.sequence_len > 1 and 'test' not in self.list_path and 'camera' not in self.list_path:
            sequence = self.files[index:index + self.sequence_len]
            if len(sequence) < self.sequence_len:
                sequence = sequence + [sequence[-1]] * (self.sequence_len - len(sequence))
        else:
            sequence = [self.files[index]]

        images = []
        labels = []
        size = None
        name = sequence[-1]["name"]

        for item in sequence:
            # Handle pre-organized sequences from list file
            if item.get("is_sequence", False) and isinstance(item["img"], list):
                # Pre-organized sequence: item["img"] is a list of frame dicts
                frames = item["img"]
            else:
                # Single frame format
                frames = [{"img": item["img"], "label": item.get("label", None)}]
            
            for frame in frames:
                image_path = os.path.join(self.root, frame["img"])
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise FileNotFoundError(f"Failed to load image: {image_path}")
                
                if size is None:
                    size = image.shape

                if 'test' in self.list_path or 'camera' in self.list_path:
                    image = self.input_transform(image)
                    image = image.transpose((2, 0, 1))
                    images.append(image.copy())
                else:
                    label_path = os.path.join(self.root, frame["label"])
                    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    
                    if label is None:
                        raise FileNotFoundError(f"Failed to load label: {label_path}")
                    
                    label = self.convert_label(label)
                    image, label = self.gen_sample(image, label,
                                                  self.multi_scale, self.flip)
                    images.append(image.copy())
                    labels.append(label.copy())

        if 'test' in self.list_path or 'camera' in self.list_path:
            if len(images) == 1:
                return images[0], np.array(size), name
            return np.stack(images, axis=0), np.array(size), name

        if len(images) == 1:
            return images[0], labels[0], np.array(size), name

        return np.stack(images, axis=0), np.stack(labels, axis=0), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = torch.from_numpy(new_img[np.newaxis,:,:,:]).float()
                new_img = self.input_transform(new_img)
                preds = self.inference(config, model, new_img.cuda(), flip)
                if config.TEST.FLIP_TEST:
                    flip_img = new_img.flip(-1)
                    flip_preds = self.inference(config, model, flip_img.cuda(), flip)
                    preds = (preds + flip_preds) / 2
            else:
                pass
            if scale <= 1.0:
                margin_top = int(self.crop_size[0]/2*self.scale_factor)
                margin_left = int(self.crop_size[1]/2*self.scale_factor)
            else:
                pass
            final_pred += preds[:,:, 0:ori_height, 0:ori_width]
        
        return final_pred
