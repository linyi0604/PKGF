from functools import total_ordering
from re import L
import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2




class DatasetProcessing_image_PK_Opencv(Dataset):
    def __init__(self, data_path, mask_path, img_filename, transform):
        self.img_path = data_path
        self.mask_path = mask_path
        self.transform = transform
        # reading img file from file
        fp = open(img_filename, 'r')

        self.img_filename = []
        self.labels = []
        self.lesion_numers = []
        for line in fp.readlines():
            filename, label, lesion_numer = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesion_numers.append(int(lesion_numer))
        fp.close()


    def __getitem__(self, index):
        name = self.img_filename[index]
        
        img = cv2.imread(self.img_path + name, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + name, cv2.COLOR_BGR2GRAY)

        # transform中 toTensor 包含转置
        T = self.transform(image=img, mask=mask)
        img, mask = T["image"], T["mask"]
        # img = img / 255
        # mask = mask.unsqueeze(0).type(torch.float)

        img = ToTensor()(Image.fromarray(img).convert("RGB"))
        mask = ToTensor()(Image.fromarray(mask).convert("L"))


        name = self.img_filename[index]
        severity = self.labels[index]
        lesion_numer = self.lesion_numers[index]

        img_pk = torch.cat([img, mask], dim=0)

        return img_pk, severity, lesion_numer

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessing_image_mask_OpenCV(Dataset):
    def __init__(self, data_path, mask_path, img_filename, transform):
        self.img_path = data_path
        self.mask_path = mask_path
        self.transform = transform
        # reading img file from file
        fp = open(img_filename, 'r')

        self.img_filename = []
        self.labels = []
        self.lesion_numers = []
        for line in fp.readlines():
            filename, label, lesion_numer = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesion_numers.append(int(lesion_numer))
        fp.close()


    def __getitem__(self, index):
        name = self.img_filename[index]
        # opencv 读取通道数在后 应转置
        img = cv2.imread(self.img_path + name, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + name, cv2.COLOR_BGR2GRAY)
        
        # transform中 toTensor 包含转置
        T = self.transform(image=img, mask=mask)
        img, mask = T["image"], T["mask"]
        # img = img / 255
        # mask = mask.unsqueeze(0).type(torch.float)

        img = ToTensor()(Image.fromarray(img).convert("RGB"))
        mask = ToTensor()(Image.fromarray(mask).convert("L"))

        name = self.img_filename[index]
        severity = self.labels[index]
        lesion_numer = self.lesion_numers[index]
        return name, img, mask, severity, lesion_numer

    def __len__(self):
        return len(self.img_filename)