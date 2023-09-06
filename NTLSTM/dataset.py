#!/usr/bin/env Python
# coding=utf-8

from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from config import cfg
import pdb


class ntlDataset(Dataset):
    def __init__(self, dataDir='/home/zlx/NTL/ntl_xian_dataset_512/', train=True, std=100.):
        self.train = train
        self.names = []
        self.dir = []
        self.rangeNum = np.zeros((6))
        self.rangeProp = np.zeros((6))
        self.dataDir = dataDir
        self.std = std
        # self.abnormalRAD = read_abnormalFile(dataDir, train=self.train)
        #self.weatherRAD = read_rainsplitFile(dataDir, train)

        if self.train:
            #pdb.set_trace()
            #self.readDir(dataDir + '/train', maxNum)
            #self.readDir('/home/zlx/train/train1', maxNum)
            self.readDir('/home/zlx/NTL/gn_attention/xian_forward/train_list.txt')
        else:
            #self.readDir(dataDir + '/test', maxNum)
            self.readDir('/home/zlx/NTL/gn_attention/xian_forward/eval_list.txt')
            #self.readDir('/home/zlx/huaweicloud/datasets_weather/final', maxNum)
            #self.readDir('/home/zlx/huaweicloud/convlstm/test', maxNum)
        #pdb.set_trace()
 
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        imageDir = self.dir[idx]
        #pdb.set_trace()
        #print(imageDir)
        images, mask = self.readImage(imageDir)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        #if self.train:
        target = images[-13:]
        return name, images[:-13], target
        #else:
        #    return mask, images, name

    def readDir(self, dataDir):
        list_read = open(dataDir).readlines()

        for line in list_read:
            names = line.strip()
            #if sonDir not in self.weatherRAD[self.weatherType]:
            #    continue
            self.names.append(names)
            sonDir = os.path.join(self.dataDir, names)
            #areaDirs = os.listdir(sonDir)
            #for areaDir in areaDirs:
            #    if areaDir not in self.weatherRAD[self.weatherType]:
            #        continue
            #    self.names.append(areaDir)
            #    areaDir = os.path.join(sonDir, areaDir)
            self.dir.append(sonDir)

    def readImage(self, imageDir):
        imgFiles = os.listdir(imageDir)
        imgFiles.sort(key=lambda x: int(x.split('China_')[-1].split('.')[0]), reverse=True)
        chnlList = []
        for imgFile in imgFiles:
            if imgFile.split('.')[-1] != 'tif':
                continue
            else:
                imgFile = os.path.join(imageDir, imgFile)
                img = cv2.imread(imgFile, -1)
                img = img/self.std
                # 
                # self.rangeNum = self.rangeNum + Statistics.cal_rangeNum(img)
                chnlList.append(img)
        # H * W * I
        image = cv2.merge(chnlList)
        # rawimg, process_img, mask = Pretreatment.removeNoise(image)
        mask = np.ones_like(img)
        image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(1)
        return image, mask
