import torch
import torch.nn as nn
from resnet import ResNet50
from torch.nn import Parameter
import math
import torch.nn.functional as F
from aspp import ASPP

class StaticModel(nn.Module):
    def __init__(self, inputnumber=3):
        super(StaticModel, self).__init__()
        self.backbone = ResNet50(3, 16)
        self.ASPP = ASPP(2048, 256, rates=[1, 6, 12, 18])
        self.convres = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv2d(320, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(None, 2, 'bilinear', True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(None, 2, 'bilinear', True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.backbone._load_pretrained_model('./resnet50-19c8e357.pth')

    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
    def forward(self, input):
        layer4_feat, layer1_feat, conv1_feat, layer2_feat, layer3_feat = self.backbone(input)

        fea = self.ASPP(layer4_feat)
        fea_out = fea
        fea = F.interpolate(fea, layer1_feat.shape[2:] , mode='bilinear', align_corners=True)
        layer1_feat = self.convres(layer1_feat)
        fea = torch.cat([fea, layer1_feat], dim=1)
        out = self.decoder(fea)
        return out


class motionTCN(nn.Module):
    def __init__(self, k):
        super(motionTCN, self).__init__()
        if k ==1:
            self.convu1 = nn.Sequential(nn.Conv3d(64, 32, (1, 1, 3), 1, padding=(0, 0, 1)), nn.BatchNorm3d(32), nn.ReLU())
            self.convu2 = nn.Sequential(nn.Conv3d(32, 32, (1, 1, 3), 1, padding=(0, 0, 2), dilation=(1, 1, 2)), nn.BatchNorm3d(32), nn.ReLU())
            self.convu3 = nn.Sequential(nn.Conv3d(32, 32, (1, 1, 3), 1, padding=(0, 0, 3), dilation=(1, 1, 3)), nn.BatchNorm3d(32), nn.ReLU())

            self.convd3 = nn.Sequential(nn.Conv3d(32, 32, (1, 1, 3), 1, padding=(0, 0, 1)), nn.BatchNorm3d(32),
                                        nn.ReLU())
            self.convd2 = nn.Sequential(nn.Conv3d(32, 32, (1, 1, 3), 1, padding=(0, 0, 2), dilation=(1, 1, 2)),
                                        nn.BatchNorm3d(32), nn.ReLU())
            self.convd1 = nn.Sequential(nn.Conv3d(64, 32, (1, 1, 3), 1, padding=(0, 0, 3), dilation=(1, 1, 3)),
                                        nn.BatchNorm3d(32), nn.ReLU())


        else:
            self.convu1 = nn.Sequential(nn.Conv3d(64, 32, (3, 3, 3), 1, padding=((k-1)/2, (k-1)/2, 1), dilation=((k-1)/2, (k-1)/2, 1)), nn.BatchNorm3d(32), nn.ReLU())
            self.convu2 = nn.Sequential(nn.Conv3d(32, 32, (3, 3, 3), 1, padding=((k-1)/2, (k-1)/2, 2), dilation=((k-1)/2, (k-1)/2, 2)), nn.BatchNorm3d(32), nn.ReLU())
            self.convu3 = nn.Sequential(nn.Conv3d(32, 32, (3, 3, 3), 1, padding=((k-1)/2, (k-1)/2, 3), dilation=((k-1)/2, (k-1)/2, 3)), nn.BatchNorm3d(32), nn.ReLU())

            self.convd3 = nn.Sequential(nn.Conv3d(32, 32, (3, 3, 3), 1, padding=((k - 1) / 2, (k - 1) / 2, 1),
                                                  dilation=((k - 1) / 2, (k - 1) / 2, 1)), nn.BatchNorm3d(32),
                                        nn.ReLU())
            self.convd2 = nn.Sequential(nn.Conv3d(32, 32, (3, 3, 3), 1, padding=((k - 1) / 2, (k - 1) / 2, 2),
                                                  dilation=((k - 1) / 2, (k - 1) / 2, 2)), nn.BatchNorm3d(32),
                                        nn.ReLU())
            self.convd1 = nn.Sequential(nn.Conv3d(64, 32, (3, 3, 3), 1, padding=((k - 1) / 2, (k - 1) / 2, 3),
                                                  dilation=((k - 1) / 2, (k - 1) / 2, 3)), nn.BatchNorm3d(32),
                                        nn.ReLU())

    def forward(self, fea):
        fea_in = fea
        feau = self.convu1(fea)
        feau = self.convu2(feau)
        feau = self.convu3(feau)
        # feau = torch.sum(feau, dim=2)

        fead = self.convd1(fea_in)
        fead = self.convd2(fead)
        fead = self.convd3(fead)
        # fead = torch.sum(fead, dim=2)

        fea_out = torch.cat([feau, fead], dim=1)
        return fea_out

class Model(nn.Module):
    def __init__(self, inputnumber):
        super(Model, self).__init__()
        self.features = StaticModel(inputnumber)
        self.motion1 = motionTCN(1)
        self.motion2 = motionTCN(3)
        self.motion3 = motionTCN(5)
        self.motion4 = motionTCN(7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, frame1, frame2, frame3, frame4, frame5):
        f1, low_level_feat1, conv1_feat1, layer2_feat1, layer3_feat1 = self.features.backbone(frame1)
        f2, low_level_feat2, conv1_feat2, layer2_feat2, layer3_feat2 = self.features.backbone(frame2)
        f3, low_level_feat3, conv1_feat3, layer2_feat3, layer3_feat3 = self.features.backbone(frame3)
        f4, low_level_feat4, conv1_feat4, layer2_feat4, layer3_feat4 = self.features.backbone(frame4)
        f5, low_level_feat5, conv1_feat5, layer2_feat5, layer3_feat5 = self.features.backbone(frame5)

        f1 = self.features.ASPP(f1)
        f2 = self.features.ASPP(f2)
        f3 = self.features.ASPP(f3)
        f4 = self.features.ASPP(f4)
        f5 = self.features.ASPP(f5)

        f11, f12, f13, f14 = f1.chunk(4, dim=1)
        f21, f22, f23, f24 = f2.chunk(4, dim=1)
        f31, f32, f33, f34 = f3.chunk(4, dim=1)
        f41, f42, f43, f44 = f4.chunk(4, dim=1)
        f51, f52, f53, f54 = f5.chunk(4, dim=1)

        f_1 = torch.cat([f11.unsqueeze(2), f21.unsqueeze(2), f31.unsqueeze(2), f41.unsqueeze(2), f51.unsqueeze(2)], dim=2)
        f_2 = torch.cat([f12.unsqueeze(2), f22.unsqueeze(2), f32.unsqueeze(2), f42.unsqueeze(2), f52.unsqueeze(2)],
                        dim=2)
        f_3 = torch.cat([f13.unsqueeze(2), f23.unsqueeze(2), f33.unsqueeze(2), f43.unsqueeze(2), f53.unsqueeze(2)],
                        dim=2)
        f_4 = torch.cat([f14.unsqueeze(2), f24.unsqueeze(2), f34.unsqueeze(2), f44.unsqueeze(2), f54.unsqueeze(2)],
                        dim=2)
        fea1 = self.motion1(f_1)
        fea2 = self.motion1(f_2)
        fea3 = self.motion1(f_3)
        fea4 = self.motion1(f_4)
        fea = torch.cat([fea1, fea2, fea3, fea4], dim=1)
        fea = 0.1 * fea[:, :, 0, :, :] + 0.2 * fea[:, :, 1, :, :] + 0.4 * fea[:, :, 2, :, :] + 0.2 * fea[:, :, 3, :, :] + 0.1 * fea[:, :, 4, :, :]
        low_level_feat3 = self.features.convres(low_level_feat3)
        fea = F.interpolate(fea, low_level_feat3.shape[2:], mode='bilinear', align_corners=True)

        feao = torch.cat([fea, low_level_feat3], dim=1)
        out = self.features.decoder(feao)

        return out




