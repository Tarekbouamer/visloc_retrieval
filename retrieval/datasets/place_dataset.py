import os

import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset


class PlaceDataset(Dataset):
    def __init__(self, root_dir, image_names, max_size=(480, 640)):
        super().__init__()

        # parse text file
        self.images, self.numDb = self.parse_text_file(image_names)

        # check if images are relative to root dir
        self.images = [os.path.join(root_dir, image) for image in self.images]

        # transform
        self.transform = transforms.Compose([
            transforms.Resize(max_size, antialias=True),
            transforms.ToTensor()
        ])

        logger.info(f'PlaceDataset with {len(self)} images')


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #
        image = Image.open(self.images[index]).convert('RGB')

        image = self.transform(image)

        return {'image': image, 'index': index}
    
    @staticmethod
    def parse_text_file(textfile):

        with open(textfile, 'r') as f:
            image_list = f.read().splitlines()

        if 'robotcar' in image_list[0].lower():
            image_list = [os.path.splitext('/'.join(q_im.split('/')[-3:]))[0] for q_im in image_list]

        num_images = len(image_list)
        
        return image_list, num_images