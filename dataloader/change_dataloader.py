import os
import random

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
import cv2
from config import opt


class Change_Detection_Train_Loader(data.Dataset):
    def __init__(self, image_before, image_after, gt_root, trainsize, advance=True,
                 color_setting=None):
        self.trainsize = trainsize

        self.images_before = [image_before + f for f in os.listdir(image_before) if
                              f.endswith('.jpg') or f.endswith('.png')]
        self.images_after = [image_after + f for f in os.listdir(image_after) if
                             f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images_before = sorted(self.images_before)
        self.images_after = sorted(self.images_after)
        self.gts = sorted(self.gts)

        self.filter_files()
        self.advance = advance

        self.size = len(self.images_before)

        self.color_setting = color_setting
        if self.color_setting is not None:
            self.color_transform = transforms.ColorJitter(brightness=float(color_setting[0]),
                                                          contrast=float(color_setting[1]),
                                                          saturation=float(color_setting[2]),
                                                          hue=float(color_setting[3]))

        self.transform = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize)),
            transforms.RandomCrop(opt.cropsize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize), interpolation=InterpolationMode.NEAREST),
            transforms.RandomCrop(opt.cropsize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform_val = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        before = self.rgb_loader(self.images_before[index])
        after = self.rgb_loader(self.images_after[index])
        gt = self.binary_loader(self.gts[index])

        seed = np.random.randint(3407)

        if self.advance:
            if self.color_setting is not None:
                before = self.color_transform(before)
                after = self.color_transform(after)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            before = self.transform(before)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            after = self.transform(after)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            gt = self.gt_transform(gt)
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            before = self.transform_val(before)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            after = self.transform_val(after)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            gt = self.gt_transform_val(gt)
        return before, after, gt

    def filter_files(self):
        assert len(self.images_before) == len(self.gts) and len(self.images_after) == len(self.gts)
        images_before = []
        images_after = []
        gts = []
        for before_path, after_path, gt_path in zip(self.images_before, self.images_after, self.gts):
            before = Image.open(before_path)
            after = Image.open(after_path)
            gt = Image.open(gt_path)
            if before.size == gt.size and after.size == gt.size:
                images_before.append(before_path)
                images_after.append(after_path)
                gts.append(gt_path)
        self.images_before = images_before
        self.images_after = images_after
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_before, image_after, gt_root, batchsize, trainsize, color_setting, shuffle=True, num_workers=4,
               pin_memory=True, advance=True):
    dataset = Change_Detection_Train_Loader(image_before, image_after, gt_root, trainsize, advance,
                                            color_setting=color_setting)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def crop_right_upper(image):
    return transforms.functional.crop(image, 32, 768, 256, 256)

class test_dataset:
    def __init__(self, image_before, image_after, gt_root, testsize):
        self.testsize = testsize

        self.images_before = [image_before + f for f in os.listdir(image_before) if
                              f.endswith('.jpg') or f.endswith('.png')]
        self.images_after = [image_after + f for f in os.listdir(image_after) if
                             f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images_before = sorted(self.images_before)
        self.images_after = sorted(self.images_after)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            # transforms.CenterCrop(512),
            # transforms.Lambda(crop_right_upper),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            # transforms.CenterCrop(512),
            # transforms.Lambda(crop_right_upper),
            transforms.ToTensor()
        ])
        self.size = len(self.images_before)
        self.index = 0

    def load_data(self):
        before = self.rgb_loader(self.images_before[self.index])
        after = self.rgb_loader(self.images_after[self.index])
        gt = self.binary_loader(self.gts[self.index])

        seed = np.random.randint(3407)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        before = self.transform(before).unsqueeze(0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        after = self.transform(after).unsqueeze(0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        gt = self.gt_transform(gt).unsqueeze(0)

        name = self.images_before[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return before, after, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
