import glob
import random
import os
import numpy as np
from PIL import Image
import math
import numpy as np
import os
import pathlib
import random
import scenenet_pb2 as sn
import sys

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


def getMeanDepth(depth_img):
    depth = depth_img.type(torch.FloatTensor)/1000
    depth = depth.view(1,-1)
    mean_depth = torch.mean(depth)
    return mean_depth


def get_Camera_parameters():
    root = '/home/shaoche/code/coop-3dod-infra/images/cameras/'
    extrinsics = []
    for camNum in range(6):
        path_extrinsic = root + 'cam'+str(camNum)+'_extrinsic.txt'
        extrinsic = np.loadtxt(path_extrinsic)
        extrinsics.append(extrinsic)
    return extrinsics


def getSingleImage(img_path, transform):
    img = Image.open(img_path)
    w, h = img.size
    img = img.crop((0, 0, w, h))
    img = transform(img)
    return img


def getImginfo(imgpath):
    info = imgpath.split('/')[-1]
    info = info.split('.')[0]
    info = info.split('_')

    cam = int(info[0])
    coordinate = [int(info[i]) for i in range(1, 5)]
    scale_u = coordinate[0] - coordinate[2]
    scale_v = coordinate[1] - coordinate[3]
    scale = np.asarray([float(scale_u), float(scale_v)])
    return cam, scale


class coopDataset(Dataset):
    def __init__(self, data_root_path, transforms_=None, phase="train"):
        self.transform = transforms.Compose(transforms_)
        f = open(data_root_path,'r')
        self.files = f.readlines()
        self.extrinsics = get_Camera_parameters()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')
        img_a = getSingleImage(img_path[0], self.transform)
        img_b = getSingleImage(img_path[1], self.transform)
        center_world = np.asarray([float(img_path[2]),float(img_path[3]),float(img_path[4])])
        img = torch.cat((img_a, img_b), dim=0)

        cam_a, scale_a = getImginfo(img_path[0])
        cam_b, scale_b = getImginfo(img_path[1])

        scale = np.vstack((scale_a, scale_b, 0.5*(scale_a+scale_b)))
        extrinsic_a = self.extrinsics[cam_a]
        extrinsic_a = extrinsic_a[:3, :]
        extrinsic_b = self.extrinsics[cam_b]
        extrinsic_b = extrinsic_b[:3, :]
        extrinsic = np.concatenate((extrinsic_a, extrinsic_b),axis=0)

        return {'img_crop': img, 'extrinsic': extrinsic, 'center': center_world,'scale': scale}

    def __len__(self):
        return len(self.files)




