import logging
import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom


def random_rot_flip(image: np.ndarray, label: np.ndarray):
    def random_rotation90(image: np.ndarray, label: np.ndarray):
        k = np.random.randint(0, 4)

        image = np.rot90(image, k, axes=(1, 2))
        label = np.rot90(label, k)

        return image, label

    def random_flip(image: np.ndarray, label: np.ndarray):
        axis = np.random.randint(0, 2)

        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return image, label

    image, label = random_rotation90(image, label)
    image, label = random_flip(image, label)

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, order=3):
        """
        order int, optional
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
        """
        self.output_size = output_size
        self.order = order

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # logging.info("%s Start Random Generator: %s", sample["idx"], image.shape)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
            # logging.info("%s random_rot_flip: %s", sample["idx"], image.shape)

        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
            # logging.info("%s random_rotate: %s", sample["idx"], image.shape)

        x, y = image.shape[-2:]
        # logging.info(
        #     "%s Random Generator zoom: %s, %s, %s, %s",
        #     sample["idx"],
        #     image.shape,
        #     x,
        #     y,
        #     self.output_size,
        # )

        if x != self.output_size[0] or y != self.output_size[1]:

            zoom_factor = (self.output_size[0] / x, self.output_size[1] / y)
            label = zoom(label, zoom_factor, order=0)

            if len(image.shape) > 2:
                zoom_factor = (1, self.output_size[0] / x, self.output_size[1] / y)

            image = zoom(image, zoom_factor, order=self.order)

        assert (image.shape[-2] == self.output_size[0]) and (
            image.shape[-1] == self.output_size[1]
        )
        image[image < 0] = 0
        image[image > 1] = 1
        image = torch.from_numpy(image.astype(np.float32))
        
        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        else:
            assert image.shape[-3] == 3, f"Oh no! This assertion failed! {image.shape}"

        label = torch.from_numpy(label.astype(np.uint8))

        sample["image"], sample["label"] = image, label
        return sample
