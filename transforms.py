import torch
import math
import numbers
import random
import numpy as np
import cv2

from PIL import Image, ImageOps

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        frame1, frame2, frame3, label = sample['frame1'], sample['frame2'], sample['frame3'], sample['label']

        if self.padding > 0:
            frame1 = ImageOps.expand(frame1, border=self.padding, fill=0)
            frame2 = ImageOps.expand(frame2, border=self.padding, fill=0)
            frame3 = ImageOps.expand(frame3, border=self.padding, fill=0)
            label = ImageOps.expand(label, border=self.padding, fill=0)


        assert frame1.size == label.size
        assert frame2.size == label.size
        w, h = frame1.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}
        if w < tw or h < th:
            frame1 = frame1.resize((tw, th), Image.BILINEAR)
            frame2 = frame2.resize((tw, th), Image.BILINEAR)
            frame3 = frame3.resize((tw, th), Image.BILINEAR)
            label = label.resize((tw, th), Image.NEAREST)

            return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        frame1 = frame1.crop((x1, y1, x1 + tw, y1 + th))
        frame2 = frame2.crop((x1, y1, x1 + tw, y1 + th))
        frame3 = frame3.crop((x1, y1, x1 + tw, y1 + th))
        label = label.crop((x1, y1, x1 + tw, y1 + th))


        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        frame1 = sample['frame1']
        frame2 = sample['frame2']
        frame3 = sample['frame3']

        label = sample['label']

        w, h = frame1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        frame1 = frame1.crop((x1, y1, x1 + tw, y1 + th))
        frame2 = frame2.crop((x1, y1, x1 + tw, y1 + th))
        frame3 = frame3.crop((x1, y1, x1 + tw, y1 + th))

        label = label.crop((x1, y1, x1 + tw, y1 + th))


        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        frame1, frame2, frame3, label = sample['frame1'], sample['frame2'], sample['frame3'], sample['label']
        if random.random() < 0.5:
            frame1 = frame1.transpose(Image.FLIP_LEFT_RIGHT)
            frame2 = frame2.transpose(Image.FLIP_LEFT_RIGHT)
            frame3 = frame3.transpose(Image.FLIP_LEFT_RIGHT)

            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        frame1 = np.array(sample['frame1']).astype(np.float32)
        frame2 = np.array(sample['frame2']).astype(np.float32)
        frame3 = np.array(sample['frame3']).astype(np.float32)
        frame4 = np.array(sample['frame4']).astype(np.float32)
        frame5 = np.array(sample['frame5']).astype(np.float32)
        label = sample['label'].astype(np.float32)

        frame1 /= 255.0
        frame1 -= self.mean
        frame1 /= self.std
        frame2 /= 255.0
        frame2 -= self.mean
        frame2 /= self.std
        frame3 /= 255.0
        frame3 -= self.mean
        frame3 /= self.std
        frame4 -= self.mean
        frame4 /= self.std
        frame5 -= self.mean
        frame5 /= self.std


        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'frame4':frame4, 'frame5':frame5, 'label':label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        frame1 = np.array(sample['frame1']).astype(np.float32).transpose((2, 0, 1))
        frame2 = np.array(sample['frame2']).astype(np.float32).transpose((2, 0, 1))
        frame3 = np.array(sample['frame3']).astype(np.float32).transpose((2, 0, 1))
        frame4 = np.array(sample['frame4']).astype(np.float32).transpose((2, 0, 1))
        frame5 = np.array(sample['frame5']).astype(np.float32).transpose((2, 0, 1))
        label = np.expand_dims(sample['label'].astype(np.float32), -1).transpose((2, 0, 1))
        label[label == 255] = 0

        frame1 = torch.from_numpy(frame1).float()
        frame2 = torch.from_numpy(frame2).float()
        frame3 = torch.from_numpy(frame3).float()
        frame4 = torch.from_numpy(frame4).float()
        frame5 = torch.from_numpy(frame5).float()
        label = torch.from_numpy(label).float()


        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'frame4':frame4, 'frame5':frame5, 'label':label}


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        frame1, frame2, frame3, frame4, frame5, label = sample['frame1'], sample['frame2'], sample['frame3'], sample['frame4'], sample['frame5'], sample['label']

        frame1 = frame1.resize(self.size, Image.BILINEAR)
        frame2 = frame2.resize(self.size, Image.BILINEAR)
        frame3 = frame3.resize(self.size, Image.BILINEAR)
        frame4 = frame4.resize(self.size, Image.BILINEAR)
        frame5 = frame5.resize(self.size, Image.BILINEAR)
        label = cv2.resize(label, self.size, cv2.INTER_NEAREST)

        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'frame4':frame4, 'frame5':frame5, 'label':label}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        frame1, frame2, frame3, label = sample['frame1'], sample['frame2'], sample['frame3'], sample['label']
        w, h = frame1.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}
        oh, ow = self.size
        frame1 = frame1.resize((ow, oh), Image.BILINEAR)
        frame2 = frame2.resize((ow, oh), Image.BILINEAR)
        frame3 = frame3.resize((ow, oh), Image.BILINEAR)
        label = label.resize((ow, oh), Image.NEAREST)

        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        frame1, frame2, frame3, label = sample['frame1'], sample['frame2'], sample['frame3'], sample['label']

        for attempt in range(10):
            area = frame1.size[0] * frame1.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= frame1.size[0] and h <= frame1.size[1]:
                x1 = random.randint(0, frame1.size[0] - w)
                y1 = random.randint(0, frame1.size[1] - h)

                frame1 = frame1.crop((x1, y1, x1 + w, y1 + h))
                frame2 = frame2.crop((x1, y1, x1 + w, y1 + h))
                frame3 = frame3.crop((x1, y1, x1 + w, y1 + h))

                label = label.crop((x1, y1, x1 + w, y1 + h))

                frame1 = frame1.resize((self.size, self.size), Image.BILINEAR)
                frame2 = frame2.resize((self.size, self.size), Image.BILINEAR)
                frame3 = frame3.resize((self.size, self.size), Image.BILINEAR)
                label = label.resize((self.size, self.size), Image.NEAREST)

                return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        frame1, frame2, frame3, label = sample['frame1'], sample['frame2'], sample['frame3'], sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        frame1 = frame1.rotate(rotate_degree, Image.BILINEAR)
        frame2 = frame2.rotate(rotate_degree, Image.BILINEAR)
        frame3 = frame3.rotate(rotate_degree, Image.BILINEAR)

        label = label.rotate(rotate_degree, Image.NEAREST)

        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}

class RandomRotateOrthogonal(object):

    def __call__(self, sample):
        frame1, frame2, frame3, label = sample['frame1'], sample['frame2'], sample['frame3'], sample['label']

        rotate_degree = random.randint(0, 3) * 90
        if rotate_degree > 0:
            frame1 = frame1.rotate(rotate_degree, Image.BILINEAR)
            frame2 = frame2.rotate(rotate_degree, Image.BILINEAR)
            frame3 = frame3.rotate(rotate_degree, Image.BILINEAR)
            label = label.rotate(rotate_degree, Image.NEAREST)

        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        frame1, frame2, frame3, label = sample['frame1'], sample['frame2'], sample['frame3'], sample['label']



        w = int(random.uniform(0.8, 2.5) * frame1.size[0])
        h = int(random.uniform(0.8, 2.5) * frame1.size[1])

        frame1 = frame1.resize((w, h), Image.BILINEAR)
        frame2 = frame2.resize((w, h), Image.BILINEAR)
        frame3 = frame3.resize((w, h), Image.BILINEAR)
        label = label.resize((w, h), Image.NEAREST)

        sample = {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}

        return self.crop(self.scale(sample))

class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        frame1, frame2, frame3, label = sample['frame1'], sample['frame2'], sample['frame3'], sample['label']


        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * frame1.size[0])
        h = int(scale * frame1.size[1])

        frame1 = frame1.resize((w, h), Image.BILINEAR)
        frame2 = frame2.resize((w, h), Image.BILINEAR)
        frame3 = frame3.resize((w, h), Image.BILINEAR)
        label = label.resize((w, h), Image.NEAREST)

        return {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'label':label}