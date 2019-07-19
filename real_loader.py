import os
from PIL import Image
import numpy as np
from os import listdir
import torch

from torchvision import transforms
import torch.utils.data as data
from tools import save_affordance_pair
from blender_loader import BlenderFolder

class KinectFolder(BlenderFolder):

    def __init__(self, root_path, include_depth,include_affordance):

        super(KinectFolder, self).__init__([root_path], include_depth)

    def image_transform(self, image):

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        transform_content = transforms.Compose([transforms.Resize((160, 320)),
                                             transforms.ToTensor()
                                             ])

        transformed = transform_content(image)
        return transformed

    def depth_transform(self, image):

        image = image.getchannel(0)

        transform_content = transforms.Compose([transforms.Resize((160, 320)),
                                             transforms.ToTensor()
                                             ])

        transformed = transform_content(image)
        transformed = transformed.float()

        return transformed

    def generate_examples(self, num_examples=10, folder_name='examples'):

        for idx in range(num_examples):
            sample, target = self.__getitem__(idx)
            img = sample[:3]
            depth = torch.unsqueeze(sample[3], 0)

            save_affordance_pair(img, target, depth,
                                 save_file=os.path.join(self.root_path, folder_name, 'pair_example_{}.jpg'.format(idx)))

class KinectEvaluationLoader(object):

    def __init__(self, include_depth, data_path=os.path.join(os.environ["HOME"],'real_images')):

        dataset = KinectFolder(data_path, include_depth,include_affordance=False)
        self.dataset = dataset

    def get(self, idx):
        sample, _ = self.dataset.__getitem__(idx)
        return torch.unsqueeze(sample, 0), None

    def get_samples(self, sample_list):

        sample, _ = self.get(sample_list[0])
        samples = torch.empty(len(sample_list), sample.shape[1], sample.shape[2], sample.shape[3])

        samples[0] = sample[0]

        for i in range(1, len(sample_list)):
            sample, _ = self.dataset.__getitem__(sample_list[i])
            samples[i] = sample

        return samples, None