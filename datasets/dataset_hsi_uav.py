import os

# import h5py
# import glob
import numpy as np
import cv2
from torch.utils.data import Dataset


class UAV_HSI_Crop_dataset(Dataset):
    def __init__(self, base_dir, list_dir=None, split="train", transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir

        self.sample_list = []
        if self.split.find("train") != -1:
            self.all_slices = os.path.join(self.data_dir, "Train/Training")

        elif self.split.find("val") != -1:
            self.all_slices = os.path.join(self.data_dir, "Train/Validation")

        elif self.split.find("test") != -1:
            self.all_slices = os.path.join(self.data_dir, "Test")

        # self.sample_list = list(glob.glob(self.all_slices + "/gt/*.npy", recursive=True))
        for root, _, fnames in os.walk(os.path.join(self.all_slices, "rs")):
            for fname in fnames:
                self.sample_list.append((fname, os.path.join(root, fname)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        (fname, fin) = self.sample_list[idx]
        image = np.load(fin)
        
        if self.split.find("train") == -1:
            print(image.dtype, image.shape, image.max(), image.min())

        # # =====================================================================
        # # Convert the image to grayscale
        # # print(image.dtype, image.shape, image.max(), image.min())
        # image = np.asarray(255 * image[..., (49, 89, 180)], dtype="uint8")
        # # print(image.dtype, image.shape, image.max(), image.min())

        # # Convert the image to grayscale
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = np.asarray(image, dtype="float32") / 255
        # # =====================================================================
        # Convert the image to RGB
        image = image[..., (49, 89, 180)]
        image = np.rollaxis(image, -1, 0)
        # =====================================================================

        fin = os.path.join(self.all_slices, "gt", fname)
        label = np.load(fin)

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)

        sample["case_name"] = str(fname)
        return sample
