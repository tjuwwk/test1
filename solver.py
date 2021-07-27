import torch
from collections import OrderedDict
from torch.optim import Adam, SGD
from model_TCN import Model, StaticModel
import torchvision.utils as vutils
import torch.nn.functional as F
import os
import numpy as np
import cv2
import torch.nn as nn


EPSILON = 1e-8
p = OrderedDict()

p['lr_bone'] = 5e-5  # Learning rate
p['lr_branch'] = 0.025
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [9, 20]
nAveGrad = 10  # Update the weights once in 'nAveGrad' forward passes
showEvery = 50
tmp_path = 'tmp_out'

class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold

        self.build_model()

        # if config.mode == 'test':
        #     self.net_bone.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 返回一个tensor变量内所有元素个数
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        print('mode: {}'.format(self.config.mode))
        print('------------------------------------------')
        if self.config.train_step == 1:
            self.net_bone = StaticModel(3)
        else:
            self.net_bone = Model(3)
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()

        if self.config.mode == 'train':
            if self.config.model_path != '':
                assert (os.path.exists(self.config.model_path)), ('please import correct pretrained model path!')
                self.net_bone.load_pretrain_model(self.config.model_path)
            if self.config.static_path != '':
                assert (os.path.exists(self.config.static_path)), ('please import correct pretrained model path!')
                self.net_bone.features.load_pretrain_model(self.config.static_path)
        else:
            assert (self.config.model_path != ''), ('Test mode, please import pretrained model path!')
            assert (os.path.exists(self.config.model_path)), ('please import correct pretrained model path!')
            self.net_bone.load_pretrain_model(self.config.model_path)

        self.lr_bone = p['lr_bone']
        self.lr_branch = p['lr_branch']
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone,
                                   weight_decay=p['wd'])
        print('------------------------------------------')
        self.print_network(self.net_bone, 'DSNet')
        print('------------------------------------------')

    def test(self):
        kk = {}
        rr = {}

        if not os.path.exists(self.save_fold):
            os.makedirs(self.save_fold)
        for i, data_batch in enumerate(self.test_loader):
            frame1, frame2, frame3, frame4, frame5, label, split, size, name = data_batch['frame1'], data_batch['frame2'], data_batch['frame3'], data_batch['frame4'], data_batch['frame5'], data_batch['label'], data_batch['split'], data_batch['size'], data_batch['name']
            dataset = data_batch['dataset']

            if self.config.cuda:
                frame1, frame2, frame3, frame4, frame5 = frame1.cuda(), frame2.cuda(), frame3.cuda(), frame4.cuda(), frame5.cuda()
            with torch.no_grad():

                pre = self.net_bone(frame1, frame2, frame3, frame4, frame5)

                for i in range(self.config.test_batch_size):

                    presavefold = os.path.join(self.save_fold, dataset[i], split[i])

                    if not os.path.exists(presavefold):
                        os.makedirs(presavefold)
                    pre1 = torch.nn.Sigmoid()(pre[i])
                    pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
                    pre1 = np.squeeze(pre1.cpu().data.numpy()) * 255
                    pre1 = cv2.resize(pre1, (size[0][1], size[0][0]))
                    cv2.imwrite(presavefold + '/' + name[i], pre1)


    def train(self):

        # 一个epoch中训练iter_num个batch
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        for epoch in range(self.config.epoch):
            r_sum_loss= 0
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):

                frame1, frame2, frame3, frame4, frame5, label = data_batch['frame1'], data_batch['frame2'], data_batch['frame3'], data_batch['frame4'], data_batch['frame5'],data_batch['label']
                if frame3.size()[2:] != label.size()[2:]:
                    print("Skip this batch")
                    continue
                if self.config.cuda:
                    frame1, frame2, frame3, frame4, frame5, label = frame1.cuda(), frame2.cuda(), frame3.cuda(), frame4.cuda(), frame5.cuda(), label.cuda()

                if self.config.train_step == 1:
                    pre1 = self.net_bone(frame1)
                else:

                    pre1 = self.net_bone(frame1, frame2, frame3, frame4, frame5)
                bce = nn.BCEWithLogitsLoss()
                # g = gloss()
                b1 = bce(pre1, label)
                # g1 = g(pre1, label)

                loss = b1
                loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0:
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()
                    aveGrad = 0

                if i % showEvery == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  Loss || sal : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num,
                        loss)  )
                    print('Learning rate: ' + str(self.lr_bone))

                if i % 50 == 0:
                    vutils.save_image(torch.sigmoid(pre1.data), tmp_path + '/iter%d-sal-0.jpg' % i,
                                      normalize=True, padding=0)
                    # vutils.save_image(torch.sigmoid(edge_out.data), tmp_path + '/iter%d-edge-0.jpg' % i,
                    #                   normalize=True, padding=0)
                    vutils.save_image(frame2.data, tmp_path + '/iter%d-sal-data.jpg' % i, padding=0)
                    vutils.save_image(label.data, tmp_path + '/iter%d-sal-target.jpg' % i, padding=0)

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           '%s/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            if epoch in lr_decay_epoch:
                self.lr_bone = self.lr_bone * 0.2
                self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()),
                                           lr=self.lr_bone, weight_decay=p['wd'])

        torch.save(self.net_bone.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)


def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    l = F.pad(x, [1, 0, 0, 0])[:, :, :, :w_x]
    t = F.pad(x, [0, 0, 1, 0])[:, :, :h_x, :]
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)

    return xgrad

class gloss(nn.Module):
    def __init__(self):
        super(gloss, self).__init__()

    def forward(self, x, gt):
        x_grad = gradient(x)
        gt_grad = gradient(gt)
        edge = torch.where(gt_grad>0, torch.ones_like(gt), torch.zeros_like(gt))
        gg = (1 - edge) * gt
        mask = torch.where(gg > 0, x_grad, torch.zeros_like(gt))
        l1 = torch.mean(mask)

        maske = torch.where(edge>0, x_grad, torch.zeros_like(gt))
        l2 = torch.exp(-torch.mean(maske))
        loss = l1*l2
        return loss


