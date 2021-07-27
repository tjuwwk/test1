import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
import transforms as transform
from torchvision import transforms

splits_DAVIS_train = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare',
                    'bus', 'car-turn', 'dance-jump', 'dog-agility',
                    'drift-turn', 'elephant', 'flamingo', 'hike',
                    'hockey', 'horsejump-low', 'kite-walk', 'lucia',
                    'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike',
                    'paragliding', 'rhino', 'rollerblade', 'scooter-gray',
                    'soccerball', 'stroller', 'surf', 'swing', 'tennis',
                    'train']
splits_DAVIS_test = ['blackswan', 'bmx-trees', 'breakdance', 'camel',
                    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl',
                    'dog', 'drift-chicane', 'drift-straight', 'goat',
                    'horsejump-high', 'kite-surf', 'libby', 'motocross-jump',
                    'paragliding-launch', 'parkour', 'scooter-black',
                    'soapbox']
splits_SegV2_test = ['bird_of_paradise', 'birdfall', 'bmx', 'cheetah', 'drift', 'frog', 'girl',
                    'hummingbird', 'monkey', 'monkeydog', 'parachute', 'soldier', 'worm']
splits_FBMS_test = ['camel01', 'cars1', 'cars10', 'cars4',
                    'cars5', 'cats01', 'cats03', 'cats06',
                    'dogs01', 'dogs02', 'farm01', 'giraffes01',
                    'goats01', 'horses02', 'horses04', 'horses05',
                    'lion01', 'marple12', 'marple2', 'marple4',
                    'marple6', 'marple7', 'marple9', 'people03',
                    'people1', 'people2', 'rabbits02', 'rabbits03',
                    'rabbits04', 'tennis']
splits_FBMS_train = ['bear01', 'bear02', 'cars2', 'cars3',
                    'cars6', 'cars7', 'cars8', 'cars9',
                    'cats02', 'cats04' ,'cats05', 'cats07',
                    'ducks01', 'horses01', 'horses03', 'horses06',
                    'lion02', 'marple1', 'marple10', 'marple11',
                    'marple13', 'marple3', 'marple5', 'marple8',
                    'meerkats01', 'people04', 'people05', 'rabbits01',
                    'rabbits05']

splits_ViSal_test = ['aeroplane', 'bird', 'boat', 'boat2', 'car', 'cat', 'cow', 'cow2', 'gokart', 'horse', 'horse2', 'lion', 'man', 'motorbike', 'panda', 'rider', 'snow_leopards']

class VideoDataset(data.Dataset):
    def __init__(self, datasets, mode='train', transform=None, return_size=True):
        self.return_size = return_size
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        for (i, dataset) in enumerate(datasets):
            data_dir = './dataset/{}'.format(dataset)
            if dataset == 'DAVSOD':
                if mode == 'train':
                    datapath = os.path.join(data_dir, 'Training Set', 'Training Set')
                    splits = os.listdir(datapath)
                    splitss = [splits]
                    datapaths = [datapath]
                    n = 1
                else:
                    datapath = os.path.join(data_dir, 'Testing Set')
                    datapath1 = os.path.join(datapath, 'Difficult-20')
                    datapath2 = os.path.join(datapath, 'Easy-35')
                    datapath3 = os.path.join(datapath, 'Normal-25')
                    datapaths = [datapath1, datapath2, datapath3]
                    splits1 = os.listdir(datapath1)
                    splits2 = os.listdir(datapath2)
                    splits3 = os.listdir(datapath3)
                    splitss = [splits1, splits2, splits3]
                    n = 3
                for j in range(n):
                    for split in splitss[j]:
                        imagespath = os.path.join(datapaths[j], split, 'Imgs')
                        images = os.listdir(imagespath)
                        images.sort()
                        gtspath = os.path.join(datapaths[j], split, 'GT_object_level')
                        gts = os.listdir(gtspath)
                        gts.sort()
                        for i in range(len(images)):
                            data = {}
                            if i == 0:
                                frame1 = os.path.join(imagespath, images[i])
                                frame2 = os.path.join(imagespath, images[i])
                                frame3 = os.path.join(imagespath, images[i])
                                frame4 = os.path.join(imagespath, images[i + 1])
                                frame5 = os.path.join(imagespath, images[i + 2])
                            elif i == 1:
                                frame1 = os.path.join(imagespath, images[i - 1])
                                frame2 = os.path.join(imagespath, images[i - 1])
                                frame3 = os.path.join(imagespath, images[i])
                                frame4 = os.path.join(imagespath, images[i + 1])
                                frame5 = os.path.join(imagespath, images[i + 2])

                            elif i == len(images) - 2:
                                frame1 = os.path.join(imagespath, images[i - 2])
                                frame2 = os.path.join(imagespath, images[i - 1])
                                frame3 = os.path.join(imagespath, images[i])
                                frame4 = os.path.join(imagespath, images[i + 1])
                                frame5 = os.path.join(imagespath, images[i + 1])
                            elif i == len(images) - 1:
                                frame1 = os.path.join(imagespath, images[i - 2])
                                frame2 = os.path.join(imagespath, images[i - 1])
                                frame3 = os.path.join(imagespath, images[i])
                                frame4 = os.path.join(imagespath, images[i])
                                frame5 = os.path.join(imagespath, images[i])
                            else:
                                frame1 = os.path.join(imagespath, images[i - 2])
                                frame2 = os.path.join(imagespath, images[i - 1])
                                frame3 = os.path.join(imagespath, images[i])
                                frame4 = os.path.join(imagespath, images[i + 1])
                                frame5 = os.path.join(imagespath, images[i + 2])

                            data['images'] = [frame1, frame2, frame3, frame4, frame5]
                            data['gt'] = os.path.join(gtspath, gts[i])
                            data['name'] = images[i]
                            data['split'] = split
                            data['dataset'] = dataset
                            self.datas_id.append(data)

            elif dataset == 'DAVIS' or dataset == 'SegV2':
                if mode == 'train' and dataset == 'DAVIS':
                    splits = splits_DAVIS_train
                elif mode == 'test' and dataset == 'DAVIS':
                    splits = splits_DAVIS_test
                elif mode == 'test' and dataset == 'SegV2':
                    splits = splits_SegV2_test
                for split in splits:
                    if dataset == 'DAVIS':
                        imagespath = os.path.join(data_dir, 'JPEGImages', '480p', split)
                        images = os.listdir(imagespath)
                        images.sort()
                        gtspath = os.path.join(data_dir, 'Annotations', '480p', split)
                        gts = os.listdir(gtspath)
                        gts.sort()
                    else:
                        imagespath = os.path.join(data_dir, 'JPEGImages', split)
                        images = os.listdir(imagespath)
                        images.sort()
                        gtspath = os.path.join(data_dir, 'Annotations', split)
                        gts = os.listdir(gtspath)
                        gts.sort()


                    for i in range(len(images)):
                        data = {}
                        if i == 0:
                            frame1 = os.path.join(imagespath, images[i])
                            frame2 = os.path.join(imagespath, images[i])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i + 1])
                            frame5 = os.path.join(imagespath, images[i + 2])
                        elif i == 1:
                            frame1 = os.path.join(imagespath, images[i - 1])
                            frame2 = os.path.join(imagespath, images[i - 1])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i + 1])
                            frame5 = os.path.join(imagespath, images[i + 2])

                        elif i == len(images) - 2:
                            frame1 = os.path.join(imagespath, images[i - 2])
                            frame2 = os.path.join(imagespath, images[i - 1])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i + 1])
                            frame5 = os.path.join(imagespath, images[i + 1])
                        elif i == len(images) - 1:
                            frame1 = os.path.join(imagespath, images[i - 2])
                            frame2 = os.path.join(imagespath, images[i - 1])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i])
                            frame5 = os.path.join(imagespath, images[i])
                        else:
                            frame1 = os.path.join(imagespath, images[i - 2])
                            frame2 = os.path.join(imagespath, images[i - 1])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i + 1])
                            frame5 = os.path.join(imagespath, images[i + 2])

                        data['images'] = [frame1, frame2, frame3, frame4, frame5]
                        data['gt'] = os.path.join(gtspath, gts[i])
                        data['split'] = split
                        data['dataset'] = dataset
                        self.datas_id.append(data)

            elif dataset == 'ViSal' or dataset == 'FBMS':
                if dataset == 'ViSal':
                    splits = splits_ViSal_test
                elif mode == 'test' and dataset == 'FBMS':
                    splits = splits_FBMS_test
                else:
                    splits = splits_FBMS_train
                for split in splits:
                    imagespath = os.path.join(data_dir, 'JPEGImages', split)
                    images = os.listdir(imagespath)

                    gtspath = os.path.join(data_dir, 'Annotations', split)
                    gts = os.listdir(gtspath)

                    for i in range(len(images)):
                        data = {}
                        if i == 0:
                            frame1 = os.path.join(imagespath, images[i])
                            frame2 = os.path.join(imagespath, images[i])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i+1])
                            frame5 = os.path.join(imagespath, images[i+2])
                        elif i == 1:
                            frame1 = os.path.join(imagespath, images[i-1])
                            frame2 = os.path.join(imagespath, images[i-1])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i+1])
                            frame5 = os.path.join(imagespath, images[i+2])

                        elif i == len(images)-2:
                            frame1 = os.path.join(imagespath, images[i-2])
                            frame2 = os.path.join(imagespath, images[i-1])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i+1])
                            frame5 = os.path.join(imagespath, images[i+1])
                        elif i == len(images)-1:
                            frame1 = os.path.join(imagespath, images[i - 2])
                            frame2 = os.path.join(imagespath, images[i-1])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i])
                            frame5 = os.path.join(imagespath, images[i])
                        else:
                            frame1 = os.path.join(imagespath, images[i - 2])
                            frame2 = os.path.join(imagespath, images[i - 1])
                            frame3 = os.path.join(imagespath, images[i])
                            frame4 = os.path.join(imagespath, images[i+1])
                            frame5 = os.path.join(imagespath, images[i+2])
                        data['images'] = [frame1, frame2, frame3, frame4, frame5]
                        data['split'] = split
                        data['dataset'] = dataset
                        for gt in gts:
                            if gt.split('.')[0] == images[i].split('.')[0]:
                                data['gt'] = os.path.join(gtspath, gt)

                        if 'gt' in data.keys():
                            self.datas_id.append(data)
                        # self.datas_id.append(data)



        self.transform = transform

    def __getitem__(self, item):

        frame1 = Image.open(self.datas_id[item]['images'][0]).convert('RGB')
        frame2 = Image.open(self.datas_id[item]['images'][1]).convert('RGB')
        frame3 = Image.open(self.datas_id[item]['images'][2]).convert('RGB')
        frame4 = Image.open(self.datas_id[item]['images'][3]).convert('RGB')
        frame5 = Image.open(self.datas_id[item]['images'][4]).convert('RGB')
        label = np.array(Image.open(self.datas_id[item]['gt']).convert('L'))

        if label.max() > 0:
            label = label / 255

        w, h = frame1.size
        size = (h, w)

        sample = {'frame1':frame1, 'frame2':frame2, 'frame3': frame3, 'frame4':frame4, 'frame5':frame5, 'label':label}

        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)

        name = self.datas_id[item]['gt'].split('/')[-1]
        sample['dataset'] = self.datas_id[item]['dataset']
        sample['split'] = self.datas_id[item]['split']
        sample['name'] = name

        return sample

    def __len__(self):
        return len(self.datas_id)

class ImageDataset(data.Dataset):
    def __init__(self, transform):
        super(ImageDataset, self).__init__()
        datapath = './dataset/DUTS-TR'
        self.imagespath = os.path.join(datapath, 'DUTS-TR-Image')
        self.gtspath = os.path.join(datapath, 'Annotations')
        self.images = os.listdir(self.imagespath)
        self.images.sort()
        self.gts = os.listdir(self.gtspath)
        self.gts.sort()
        self.transform = transform

    def __len__(self):
        return len(self.gts)
    def __getitem__(self, item):
        imagepath = os.path.join(self.imagespath, self.images[item])
        gtpath = os.path.join(self.gtspath, self.gts[item])
        image = Image.open(imagepath).convert('RGB')
        label = np.array(Image.open(gtpath).convert('L'))
        if label.max() > 0:
            label = label / 255

        w, h = image.size
        size = (h, w)

        sample = {'frame1':image, 'frame2':image, 'frame3': image, 'frame4':image, 'frame5':image, 'label':label}
        if self.transform:
            sample = self.transform(sample)

        sample['size'] = torch.tensor(size)

        name = self.images[item]
        sample['dataset'] = 'DUTS-TR'
        sample['split'] = 'DUTS-TR'
        sample['name'] = name
        return sample