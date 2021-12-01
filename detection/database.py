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


class CustomImageDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args: csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.annotations.iloc[idx, 0])
        image = io.imread(img_name)
        value = self.annotations.iloc[idx, 1]
        # sample = {'image': image, 'value': value}
        sample = (image, value)

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, value = sample['image'], sample['value']
        image, value = sample

        image = image/255
        # print(image.shape)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        # return {'image': torch.from_numpy(image),
        #         'value': torch.tensor(value)}
        return (torch.from_numpy(image).double(), torch.tensor(value))




def make_label_csv_files(path, split, filename):
    # randomly create 2 label csv files with split
    image_array = [("image_name", "value")]
    count = 1
    for root, dirs, files in os.walk(path):
        for name in files:
            fullname = os.path.join(root, name)
            value = 0
            if fullname.endswith("bmp"):
                value = 1
            insert = (fullname, value)
            image_array.append(insert)
            print(f"file {count}: {fullname}")
            count = count + 1
    split_num = int(len(image_array)*split)
    train_name = f"{filename}_train.csv"
    test_name = f"{filename}_test.csv"
    savetxt(train_name, np.array(image_array[:split_num]), delimiter=',', fmt='%s')
    savetxt(test_name, np.array(image_array[split_num:]), delimiter=',', fmt='%s')
    return (train_name, test_name)

# make_label_csv_file("/home/lg08/main/data/standard_collective")


def check_size(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            fullname = os.path.join(root, name)
            image = PIL.Image.open(fullname)
            width, height = image.size
            if (width, height) != (256, 256):
                print("Wrong image size!!!!!!!!!!!!")
                return False
    return True

# check_size("/home/lg08/main/data/standard_collective")
