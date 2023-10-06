
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

_EXT = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']


class ImagesListDataset(Dataset):

    def __init__(self, images_path, cameras_path=None, split='', max_size=1024):

        self.images_path = images_path
        self.cameras_path = cameras_path
        self.split = split
        self.max_size = max_size

        # Load images
        paths = []
        for ext in _EXT:
            paths += list(Path(self.images_path).glob('**/' + ext))

        #
        if len(paths) == 0:
            raise ValueError(
                f'Could not find any image in path: {self.images_path}.')

        #
        self.images_fn = sorted(list(set(paths)))
        logger.info(
            f'found {len(self.images_fn)} images in {self.images_path}')

    def __len__(self):
        return len(self.images_fn)

    def resize_image(self, image, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return image

    def get_names(self):
        return [self.split + "/" + str(p.relative_to(self.images_path)) for p in self.images_fn]

    def load_img(self, img_path, mode='color'):
        if mode == 'gray':
            mode = cv2.IMREAD_GRAYSCALE
        elif mode == 'color':
            mode = cv2.IMREAD_COLOR
        else:
            raise KeyError(f'mode {mode} not found!')

        # read
        image = cv2.imread(str(img_path), mode)
        image = image.astype(np.float32)

        # HxWxC to CxHxW
        image = image.transpose((2, 0, 1))
        image = image / 255.
        return image

    def get_cameras(self,):
        if hasattr(self, "cameras"):
            return self.cameras
        else:
            None

    def __getitem__(self, item):
        #
        out = {}

        # load
        item_path = self.images_fn[item]
        image = self.load_img(item_path)
        ori_size = np.array(image.shape[1:])

        # dict
        out["image"] = image
        out["img_name"] = str(item_path.relative_to(
            self.images_path)).split('.')[0]
        out["original_size"] = ori_size

        return out
