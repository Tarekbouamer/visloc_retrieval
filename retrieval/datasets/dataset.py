from os import path
from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

INPUTS = ["image"]


_EXT = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagesFromList(Dataset):
    """A generic dataset from a list of images"""

    def __init__(self, data_path, images_names=None, bbxs=None, transform=None):

        images = []

        if images_names is not None:
            images = [path.join(data_path, images_names[i])
                      for i in range(len(images_names))]
        else:
            for ext in _EXT:
                images += list(Path(data_path).glob('**/' + ext))

        # check if images are present
        assert len(images) > 0, f'No images found in {data_path}'

        # data path
        self.data_path = data_path

        # images
        self.images = images

        # bbxs
        self.bbxs = bbxs

        # transform
        self.transform = transform

    def __len__(self):
        """Return the number of images"""
        return len(self.images)

    def load_img(self, img_path):
        """Load image"""

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Cannot load image {img_path}")
            raise e

        return image

    def __getitem__(self, item):

        # image path and image name
        img_path = self.images[item]
        img_name = path.basename(img_path)

        # load image
        image = self.load_img(img_path)

        # crop image if box exsists
        if self.bbxs is not None:
            image = image.crop(self.bbxs[item])

        # transform
        if self.transform is not None:
            out = self.transform(image)

        # output
        out['name'] = img_name

        #
        return out
