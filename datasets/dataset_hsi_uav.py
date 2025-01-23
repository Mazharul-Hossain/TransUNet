import os
import logging
import pandas as pd
import numpy as np
import torch
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

        # if self.split.find("train") != -1:
        #     logging.info(
        #         "%s %s %s %s %s",
        #         idx,
        #         image.dtype,
        #         image.shape,
        #         image.max(),
        #         image.min(),
        #     )

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
        # image = image[..., (49, 89, 180)]
        image = self._get_rgb_image(image)
        image[image < 0] = 0
        image[image > 1] = 1

        # logging.info("Before roll axis: %s %s", idx, image.shape)
        image = np.rollaxis(image, -1)
        # =====================================================================

        fin = os.path.join(self.all_slices, "gt", fname)
        label = np.load(fin)

        sample = {"image": image, "label": label, "idx": idx}
        # if self.split.find("train") != -1:
        #     logging.info(
        #         "UAV_HSI_Crop Dataset generator: %s %s %s %s %s",
        #         idx,
        #         sample["image"].dtype,
        #         sample["image"].shape,
        #         sample["image"].max(),
        #         sample["image"].min(),
        #     )

        if self.transform:
            sample = self.transform(sample)

        else:            
            sample["image"] = torch.from_numpy(sample["image"].astype(np.float32))
            
            if len(sample["image"].shape) == 2:
                sample["image"] = sample["image"].unsqueeze(0)

            else:
                assert sample["image"].shape[-3] == 3, f"Oh no! This assertion failed! {sample["image"].shape}"

            sample["label"] = torch.from_numpy(sample["label"].astype(np.uint8))

        sample["case_name"] = str(fname)
        return sample

    def _get_rgb_image(self, hsi):
        band_dictionary = {
            "visible-violet": {"lower": 365, "upper": 450, "color": "violet"},
            "visible-blue": {
                "lower": 450,
                "upper": 485,
                "color": "blue",
            },  # BlueWavelengths = 450:495
            "visible-cyan": {"lower": 485, "upper": 500, "color": "cyan"},
            "visible-green": {
                "lower": 500,
                "upper": 565,
                "color": "green",
            },  # GreenWavelengths = 550:570
            "visible-yellow": {"lower": 565, "upper": 590, "color": "yellow"},
            "visible-orange": {"lower": 590, "upper": 625, "color": "orange"},
            "visible-red": {
                "lower": 625,
                "upper": 740,
                "color": "red",
            },  # RedWavelengths = 620:659
            "near-infrared": {"lower": 740, "upper": 1100, "color": "gray"},
            "shortwave-infrared": {"lower": 1100, "upper": 2500, "color": "white"},
        }

        def classifier(band):
            # function to classify bands
            def between(wavelength, region):
                return region["lower"] < wavelength <= region["upper"]

            for region, limits in band_dictionary.items():
                if between(band, limits):
                    return region

        def get_band_centers(band_centers):
            # print(band_centers)

            band_numbers = [i for i in range(1, len(band_centers) + 1)]
            # print(band_numbers)

            em_regions = [classifier(b) for b in band_centers]
            # print(em_regions)

            return band_centers, band_numbers, em_regions

        def get_all_bands(wav):
            band_centers, band_numbers, em_regions = get_band_centers(wav)

            # data frame describing bands
            bands = pd.DataFrame(
                {
                    "Band number": band_numbers,
                    "Band center (nm)": band_centers,
                    "EM region": em_regions,
                },
                index=band_numbers,
            ).sort_index()

            return bands

        def get_local_bands(hsi, bands, start, end):
            def get_band_number(w, bands):
                return bands.iloc[(bands["Band center (nm)"] - w).abs().argsort()[1]]

            Si = get_band_number(start, bands)
            Ei = get_band_number(end, bands)

            # # print band information from the table
            # print("---" * 20)
            # print(str("\n" + "---" * 20 + "\n").join([str(Si), str(Ei)]))

            Si, Ei = int(Si["Band number"]) - 1, int(Ei["Band number"]) - 1
            new_stack = np.zeros((nrows, ncols, Ei + 1 - Si))

            for i in range(Ei + 1 - Si):
                Sa = hsi[..., Si + i]
                Sa[Sa < 0] = 0

                new_stack[..., i] = Sa

            new_stack = np.mean(new_stack, axis=2)
            # logging.info("get_local_bands %s %s %s", new_stack.shape, nrows, ncols)

            return new_stack

        nrows, ncols, _ = hsi.shape
        wavelength = list(np.linspace(400, 1000, 200))
        bands = get_all_bands(wavelength)

        # make rgb stack
        rgb_stack = np.zeros((nrows, ncols, 3), "uint8")

        rgb_stack[..., 0], rgb_stack[..., 1], rgb_stack[..., 2] = (
            get_local_bands(
                hsi,
                bands,
                band_dictionary["visible-blue"]["lower"],
                band_dictionary["visible-blue"]["upper"],
            ),
            get_local_bands(
                hsi,
                bands,
                band_dictionary["visible-green"]["lower"],
                band_dictionary["visible-green"]["upper"],
            ),
            get_local_bands(
                hsi,
                bands,
                band_dictionary["visible-red"]["lower"],
                band_dictionary["visible-red"]["upper"],
            ),
        )

        return rgb_stack
