import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import glob
import os
from PIL import Image

'''
require fix the random seeds
'''
# PotsdamDataset
rgb_means = [0.3366, 0.3599, 0.3333]
rgb_stds = [0.1030, 0.1031, 0.1066]

class PotsdamDataset(Dataset):
    def __init__(self,
                 configs,
                 subset):
        super(PotsdamDataset, self).__init__()
        assert subset == 'train' or subset == 'valid' or subset == 'test', \
            'subset should be in the set of train, valid, and test'
        self.configs = configs
        self.subset = subset

        # sorted here when the images is not in order
        self.path_images = sorted(glob.glob(os.path.join(configs.path_cropped_images, subset, '*')))
        #print(self.path_images[0:10])
        self.path_labels = sorted(glob.glob(os.path.join(configs.path_cropped_labels, subset, '*')))
        #print(self.path_labels[0:10])

        # mapping to HxWx1
        self.mask_mapping = {
            (255, 255, 255) : 0,  # impervious surfaces
            (0, 0, 255) : 1,      # Buildings
            (0, 255, 255) : 2,    # Low Vegetation
            (0, 255, 0) : 3,      # Tree
            (255, 255, 0) : 4,    # Car
            (255, 0, 0) : 5       # background/Clutter
        }

    def mask_to_class(self, mask):
        for k in self.mask_mapping:
            # all used in numpy: axis, when used in torch: dim
            mask[(mask == torch.tensor(k, dtype = torch.uint8)).all(dim = 2)] = self.mask_mapping[k]
        return mask[:, :, 0]

    def transformation(self, image, mask):

        if self.subset == 'train':
            # only used for training data
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() > 0.5:
                image = TF.rotate(image, 10)
                mask = TF.rotate(mask, 10)

            if random.random() > 0.5:
                image = TF.rotate(image, -10)
                mask = TF.rotate(mask, -10)


        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=rgb_means, std=rgb_stds)
        # here mask's dtype uint8, the dtype of k also should be uint8 and should be a tensor
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        mask = self.mask_to_class(mask)
        mask = mask.long()
        return image, mask

    def __getitem__(self, item):

        image = Image.open(self.path_images[item])
        label = Image.open(self.path_labels[item])

        image, label = self.transformation(image, label)

        # only need filename in testing phase, used in prediction file.
        if self.subset == 'test':
            return image, label, self.path_images[item]
        else:
            return image, label

    def __len__(self):

        return len(self.path_images)

