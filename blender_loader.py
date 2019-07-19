import os
from PIL import Image
import numpy as np
from os import listdir
import torch

from torchvision import transforms
import torch.utils.data as data
from tools import save_affordance_pair


class BlenderFolder(data.Dataset):

    def __init__(self, root_paths, include_depth, debug=False):

        self.images = []
        self.depths = []
        self.affordances = []

        self.include_depth = include_depth

        for root_path in root_paths:

            assert(os.path.exists(root_path))
            images = self.list_file_paths(root_path, 'images')

            self.images.append(images)
            if self.include_depth:
                self.depths.append(self.list_file_paths(root_path, 'depths'))

            self.affordances.append(self.list_file_paths(root_path, 'affordances'))

        self.images = np.concatenate(self.images)
        if self.include_depth:
            self.depths = np.concatenate(self.depths)
        self.affordances = np.concatenate(self.affordances)

        if debug:

            self.images = self.images[-50:]
            if self.include_depth:
                self.depths = self.depths[-50:]
            self.affordances = self.affordances[-50:]

    def __getitem__(self, index):

        img_path = self.images[index]
        img = self.loader(img_path)
        img = self.image_transform(img)

        if self.include_depth:
            depth_path = self.depths[index]
            depth = self.loader(depth_path)
            depth = self.depth_transform(depth)
            sample = self.create_sample(img, depth)

        else:
            sample = img

        affordance_path = self.affordances[index]
        affordance = self.loader(affordance_path)
        target = self.affordance_transform(affordance)

        return sample, target

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        print(fmt_str)

    def create_sample(self, img, depth):
        return torch.cat([img, depth])

    def list_file_paths(self, root_path, folder_name):

        folder_path = os.path.join(root_path, folder_name)
        files = listdir(folder_path)
        files = sorted(files)
        file_paths = [os.path.join(folder_path, file_name)  for file_name in files]

        return np.array(file_paths)

    def loader(self, path):
        image = Image.open(path)
        image.load()
        return image

    def affordance_transform(self, image):

        image = np.array(image)
        affordance_tensor = np.zeros((2, image.shape[0], image.shape[1]))

        color_label= [188, 0] # 188: hole, 0: grasp

        for idx, label  in enumerate(color_label):

            indices = image[:, :, 0] == label
            affordance_tensor[idx, indices] = 1.

        affordance_torch = torch.from_numpy(affordance_tensor)
        affordance_torch = affordance_torch.float()

        return affordance_torch


    def image_transform(self, image):

        if image.mode == 'RGBA':
            image = image.convert('RGB')
        

        
        transform_content = transforms.Compose([transforms.Resize((240,320)),transforms.CenterCrop((160,320)),transforms.ToTensor()])
        transformed = transform_content(image)

        return transformed

    def depth_transform(self, image):

        image = image.getchannel(0)

        transform_content = transforms.Compose([transforms.ToTensor()])
        transformed = transform_content(image)

        transformed  = transformed.max() - transformed / (transformed.max() - transformed.min())

        return transformed

    def generate_examples(self, num_examples=10, folder_name='examples'):

        for idx in range(num_examples):
            sample, target = self.__getitem__(idx)
            img = sample[:3]
            depth = torch.unsqueeze(sample[3], 0)

            save_affordance_pair(img, target, depth,
                                 save_file=os.path.join(self.root_path, folder_name, 'pair_example_{}.jpg'.format(idx)))


class BlenderLoader(object):

    def __init__(self, batch_size, num_processes, include_depth, data_paths, debug=False, split=True):

        self.batch_size = batch_size
        self.num_processes = num_processes
        self.include_depth = include_depth

        dataset = BlenderFolder(data_paths, self.include_depth, debug=debug)
        if split:
            train_size = int(dataset.__len__() * 0.7)
            test_size = dataset.__len__() - train_size
            trainset, testset = torch.utils.data.random_split(dataset, (train_size, test_size))
        else:
            trainset = dataset
            testset = dataset
        self.trainset = trainset
        self.testset = testset

    def get_iterator(self, train):

        if train:
            dataset = self.trainset

        else:
            dataset = self.testset

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_processes)

