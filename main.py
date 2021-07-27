
import argparse
import os
from dataset import VideoDataset, ImageDataset
import torch
from solver import Solver
from torchvision import transforms
import transforms as transform
from torch.utils import data


def main(config):
    composed_transforms_ts = transforms.Compose([
        transform.FixedResize(size=(config.input_size, config.input_size)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])
    if config.mode == 'train':
        if config.train_step == 1:
            dataset = ImageDataset(transform=composed_transforms_ts)
            train_loader = data.DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_thread,
                                           drop_last=True, shuffle=True)
        else:
            dataset = VideoDataset(datasets=['DAVSOD', 'DAVIS'], transform=composed_transforms_ts, mode='train')
            train_loader = data.DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_thread, drop_last=True, shuffle=True)

        if not os.path.exists("%s/%s" % (config.save_fold, 'models')):
            os.mkdir("%s/%s" % (config.save_fold, 'models'))
        config.save_fold = "%s/%s" % (config.save_fold, 'models')
        train = Solver(train_loader, None, config)
        train.train()

    elif config.mode == 'test':

        dataset = VideoDataset(datasets=config.test_dataset, transform=composed_transforms_ts, mode='test')

        test_loader = data.DataLoader(dataset, batch_size=config.test_batch_size, num_workers=config.num_thread, drop_last=True, shuffle=False)
        test = Solver(train_loader=None, test_loader=test_loader, config=config, save_fold=config.testsavefold)
        test.test()


    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    parser = argparse.ArgumentParser()

    # Hyper-parameters11111111111111111111
    print(torch.cuda.is_available())

    parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda

    # train
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--test_batch_size', type=int, default=2)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--static_path', type=str, default='./static_384.pth')

    parser.add_argument('--save_fold', type=str, default='./result')  # 训练过程中输出的保存路径
    parser.add_argument('--input_size', type=int, default=384)
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--train_step', type=int, default=2)
    # test
    parser.add_argument('--test_dataset', type=list, default=['SegV2'])
    parser.add_argument('--testsavefold', type=str, default='./prediction')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    config = parser.parse_args()

    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
