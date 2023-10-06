import torchvision.transforms as transforms
from PIL import Image


class ImagesTransform:

    def __init__(self, max_size, preprocessing=None, augmentation=None):

        # max size
        self.max_size = max_size

        # preprocessing
        if preprocessing:
            self.preprocessing = preprocessing

        # augmentation
        if augmentation:
            self.augmentation = augmentation

        # to tensor
        self.postprocessing = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, image):

        #
        if hasattr(self, 'preprocessing'):
            image = self.preprocessing(image)
        else:
            image.thumbnail((self.max_size, self.max_size), Image.BILINEAR)

        #
        if hasattr(self, 'augmentation'):
            image = self.augmentation(image)

        # to tensor
        image = self.postprocessing(image)

        return dict(image=image)
