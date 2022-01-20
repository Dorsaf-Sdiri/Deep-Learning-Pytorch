import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
# Indexes of 100 most important features among ResNet50 feature extraction in order to reduce the Runtime
with open('Important_features.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
TRAIN_IMG_DIR = 'data\\train_input\\resnet_features\\'


class Camelyon16(Dataset):

    def __init__(self, df, transform=None):
        self.image_ids = df.ID.values
        self.ID = df.ID.values
        self.Target = df.Target.values
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        fn_base = f'{TRAIN_IMG_DIR}{self.image_ids[index]}'
        # Read npy array :
        img = np.load(f"{fn_base}.npy")
        img = img[:, 3:]
        # Dimensionality reduction using feature importance / It is possible also to use PCA or auto-encoders
        img = img[:, mynewlist]
        img = np.pad(img, [(0, 1000 - len(img)), (0, 0)], mode='constant')
        img = img[(..., *([np.newaxis] * 2))]
        label = self.Target[index]
        #         return  torch.tensor(img.transpose(0, 3, 2, 1)), torch.tensor(label)
        return torch.tensor(img), torch.tensor(label)