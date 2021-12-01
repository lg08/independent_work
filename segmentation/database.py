import os
from PIL import Image
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from numpy import savetxt, loadtxt
import numpy as np
import pandas as pd
from skimage import io, transform
import torch
import cv2

import PIL


class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.maskPaths[idx], 0)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)

