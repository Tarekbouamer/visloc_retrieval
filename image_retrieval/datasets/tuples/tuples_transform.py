# import random

# import numpy as np
# import torch
# from PIL import Image
# from torchvision.transforms import functional as tfn
# import torchvision.transforms as transforms


# class TuplesTransform:

#     def __init__(self,
#                  shortest_size=None,
#                  longest_max_size=None,
#                  rgb_mean=None,
#                  rgb_std=None,
#                  random_flip=False,
#                  random_scale=None):

#         self.shortest_size = shortest_size
#         self.longest_max_size = longest_max_size
#         self.rgb_mean = rgb_mean
#         self.rgb_std = rgb_std
#         self.random_flip = random_flip
#         self.random_scale = random_scale
        
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=rgb_mean, std=rgb_std)
#         ])

#     def _adjusted_scale(self, in_width, in_height, target_size):
        
#         max_size = max(in_width, in_height)
        
#         scale = target_size / max_size
        
#         return scale

#     @staticmethod
#     def _random_flip(img, ):
#         if random.random() < 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             return img
#         else:
#             return img

#     def _random_target_size(self):
#         if len(self.random_scale) == 2:
#             target_size = random.uniform(self.shortest_size * self.random_scale[0],
#                                          self.shortest_size * self.random_scale[1])
#         else:
#             target_sizes = [self.shortest_size * scale for scale in self.random_scale]
#             target_size = random.choice(target_sizes)
#         return int(target_size)
    
#     def _normalize_image(self, img):
        
#         if self.rgb_mean is not None:
#             img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        
#         if self.rgb_std is not None:
#             img.div_(img.new(self.rgb_std).view(-1, 1, 1))
#         return img
    
#     def __call__(self, img):
        
#         # Random flip
#         if self.random_flip:
#             img = self._random_flip(img)

#         # resize
#         img.thumbnail((self.longest_max_size, self.longest_max_size), Image.BILINEAR)

#         # Image transformations
#         img = self.transform(img)
        
#         return dict(img=img)
