import torch
import torch.nn as nn
from resnet import ResNet50
from torch.nn import Parameter
import math
import torch.nn.functional as F
from aspp import ASPP
from scipy import sparse

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)

        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

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


class Model(nn.Module):
    def __init__(self, inputnumber=3):
        super(Model, self).__init__()
        self.features = StaticModel(inputnumber)


        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.gcu11 = GraphConvolution(256, 512)
        self.gcu12 = GraphConvolution(512, 256)

        self.gcu21 = GraphConvolution(256, 512)
        self.gcu22 = GraphConvolution(512, 256)

        self.gcu31 = GraphConvolution(256, 512)
        self.gcu32 = GraphConvolution(512, 256)

        self.gcd11 = GraphConvolution(256, 512)
        self.gcd12 = GraphConvolution(512, 256)

        self.gcd21 = GraphConvolution(256, 512)
        self.gcd22 = GraphConvolution(512, 256)

        self.gcd31 = GraphConvolution(256, 512)
        self.gcd32 = GraphConvolution(512, 256)

        self.gcu111 = GraphConvolution(256, 512)
        self.gcu112 = GraphConvolution(512, 256)

        self.gcu121 = GraphConvolution(256, 512)
        self.gcu122 = GraphConvolution(512, 256)

        self.gcu131 = GraphConvolution(256, 512)
        self.gcu132 = GraphConvolution(512, 256)

        self.gcd111 = GraphConvolution(256, 512)
        self.gcd112 = GraphConvolution(512, 256)

        self.gcd121 = GraphConvolution(256, 512)
        self.gcd122 = GraphConvolution(512, 256)

        self.gcd131 = GraphConvolution(256, 512)
        self.gcd132 = GraphConvolution(512, 256)

        self.convf = nn.Sequential(nn.Conv2d(256, 256, 3, 1,1), nn.BatchNorm2d(256), nn.ReLU())
        self.convgc = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.convgc1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())


        A1 = torch.from_numpy(sparse.load_npz('A_1.npz').A).unsqueeze(0)
        self.A1 = A1.cuda()
        A3 = torch.from_numpy(sparse.load_npz('A_3.npz').A).unsqueeze(0)
        self.A3 = A3.cuda()
        A11 = torch.from_numpy(sparse.load_npz('A_1_1.npz').A).unsqueeze(0)
        self.A11 = A11.cuda()
        A31 = torch.from_numpy(sparse.load_npz('A_3_1.npz').A).unsqueeze(0)
        self.A31 = A31.cuda()


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

        fea_gc1 = f3.reshape(f3.shape[0], f3.shape[1], -1).transpose(1, 2)

        fea_gcu = self.gcu11(fea_gc1, self.A1)
        fea_gcu = self.relu(fea_gcu)
        fea_gcu = self.gcu12(fea_gcu, self.A1)
        fea_gcu = self.relu(fea_gcu)
        fea_gcu = fea_gcu.transpose(1, 2).reshape(f3.shape)

        fea_gcu = torch.cat([f2.unsqueeze(2), fea_gcu.unsqueeze(2), f4.unsqueeze(2)], dim=2)
        fea_gcuo1 = fea_gcu
        fea_gcu = fea_gcu.reshape(fea_gcu.shape[0], fea_gcu.shape[1], -1).transpose(1, 2)
        fea_gcu = self.gcu21(fea_gcu, self.A3)
        fea_gcu = self.relu(fea_gcu)
        fea_gcu = self.gcu22(fea_gcu, self.A3)
        fea_gcu = self.relu(fea_gcu)

        fea_gcu = fea_gcu.transpose(1, 2).reshape(fea_gcuo1.shape)
        fea_gcu = fea_gcu[:, :, 1, :, :]

        fea_gcu = torch.cat([f1.unsqueeze(2), fea_gcu.unsqueeze(2), f5.unsqueeze(2)], dim=2)
        fea_gcuo2 = fea_gcu
        fea_gcu = fea_gcu.reshape(fea_gcu.shape[0], fea_gcu.shape[1], -1).transpose(1, 2)
        fea_gcu = self.gcu31(fea_gcu, self.A3)
        fea_gcu = self.relu(fea_gcu)
        fea_gcu = self.gcu32(fea_gcu, self.A3)
        fea_gcu = self.relu(fea_gcu)
        fea_gcu = fea_gcu.transpose(1, 2).reshape(fea_gcuo2.shape)
        fea_gcu = fea_gcu[:, :, 1, :, :]


        fea_gc2 = torch.cat([f1.unsqueeze(2), f3.unsqueeze(2), f5.unsqueeze(2)], dim=2)
        fea_gcdo1 = fea_gc2
        fea_gc2 = fea_gc2.reshape(fea_gc2.shape[0], fea_gc2.shape[1], -1).transpose(1, 2)

        fea_gcd = self.gcd11(fea_gc2, self.A3)
        fea_gcd = self.relu(fea_gcd)
        fea_gcd = self.gcd12(fea_gcd, self.A3)
        fea_gcd = self.relu(fea_gcd)
        fea_gcd = fea_gcd.transpose(1, 2).reshape(fea_gcdo1.shape)
        fea_gcd = fea_gcd[:, :, 1, :, :]

        fea_gcd = torch.cat([f2.unsqueeze(2), fea_gcd.unsqueeze(2), f4.unsqueeze(2)], dim=2)
        fea_gcdo2 = fea_gcd
        fea_gcd = fea_gcd.reshape(fea_gcd.shape[0], fea_gcd.shape[1], -1).transpose(1, 2)
        fea_gcd = self.gcd21(fea_gcd, self.A3)
        fea_gcd = self.relu(fea_gcd)
        fea_gcd = self.gcd22(fea_gcd, self.A3)
        fea_gcd = self.relu(fea_gcd)

        fea_gcd = fea_gcd.transpose(1, 2).reshape(fea_gcdo2.shape)
        fea_gcd = fea_gcd[:, :, 1, :, :]
        fea_gcdo3 = fea_gcd

        fea_gcd = fea_gcd.reshape(fea_gcd.shape[0], fea_gcd.shape[1], -1).transpose(1, 2)
        fea_gcd = self.gcd31(fea_gcd, self.A1)
        fea_gcd = self.relu(fea_gcd)
        fea_gcd = self.gcd32(fea_gcd, self.A1)
        fea_gcd = self.relu(fea_gcd)
        fea_gcd = fea_gcd.transpose(1, 2).reshape(fea_gcdo3.shape)

        fea_gc11 = f3.reshape(f3.shape[0], f3.shape[1], -1).transpose(1, 2)

        fea_gcu1 = self.gcu111(fea_gc11, self.A11)
        fea_gcu1 = self.relu(fea_gcu1)
        fea_gcu1 = self.gcu112(fea_gcu1, self.A11)
        fea_gcu1 = self.relu(fea_gcu1)
        fea_gcu1 = fea_gcu1.transpose(1, 2).reshape(f3.shape)

        fea_gcu1 = torch.cat([f2.unsqueeze(2), fea_gcu1.unsqueeze(2), f4.unsqueeze(2)], dim=2)
        fea_gcuo11 = fea_gcu1
        fea_gcu1 = fea_gcu1.reshape(fea_gcu1.shape[0], fea_gcu1.shape[1], -1).transpose(1, 2)
        fea_gcu1 = self.gcu121(fea_gcu1, self.A31)
        fea_gcu1 = self.relu(fea_gcu1)
        fea_gcu1 = self.gcu122(fea_gcu1, self.A31)
        fea_gcu1 = self.relu(fea_gcu1)

        fea_gcu1 = fea_gcu1.transpose(1, 2).reshape(fea_gcuo11.shape)
        fea_gcu1 = fea_gcu1[:, :, 1, :, :]

        fea_gcu1 = torch.cat([f1.unsqueeze(2), fea_gcu1.unsqueeze(2), f5.unsqueeze(2)], dim=2)
        fea_gcuo12 = fea_gcu1
        fea_gcu1 = fea_gcu1.reshape(fea_gcu1.shape[0], fea_gcu1.shape[1], -1).transpose(1, 2)
        fea_gcu1 = self.gcu131(fea_gcu1, self.A31)
        fea_gcu1 = self.relu(fea_gcu1)
        fea_gcu1 = self.gcu132(fea_gcu1, self.A31)
        fea_gcu1 = self.relu(fea_gcu1)
        fea_gcu1 = fea_gcu1.transpose(1, 2).reshape(fea_gcuo12.shape)
        fea_gcu1 = fea_gcu1[:, :, 1, :, :]

        fea_gc12 = torch.cat([f1.unsqueeze(2), f3.unsqueeze(2), f5.unsqueeze(2)], dim=2)
        fea_gcdo11 = fea_gc12
        fea_gc12 = fea_gc12.reshape(fea_gc12.shape[0], fea_gc12.shape[1], -1).transpose(1, 2)

        fea_gcd1 = self.gcd111(fea_gc12, self.A31)
        fea_gcd1 = self.relu(fea_gcd1)
        fea_gcd1 = self.gcd112(fea_gcd1, self.A31)
        fea_gcd1 = self.relu(fea_gcd1)
        fea_gcd1 = fea_gcd1.transpose(1, 2).reshape(fea_gcdo11.shape)
        fea_gcd1 = fea_gcd1[:, :, 1, :, :]

        fea_gcd1 = torch.cat([f2.unsqueeze(2), fea_gcd1.unsqueeze(2), f4.unsqueeze(2)], dim=2)
        fea_gcdo12 = fea_gcd1
        fea_gcd1 = fea_gcd1.reshape(fea_gcd1.shape[0], fea_gcd1.shape[1], -1).transpose(1, 2)
        fea_gcd1 = self.gcd121(fea_gcd1, self.A31)
        fea_gcd1 = self.relu(fea_gcd1)
        fea_gcd1 = self.gcd122(fea_gcd1, self.A31)
        fea_gcd1 = self.relu(fea_gcd1)

        fea_gcd1 = fea_gcd1.transpose(1, 2).reshape(fea_gcdo12.shape)
        fea_gcd1 = fea_gcd1[:, :, 1, :, :]
        fea_gcdo13 = fea_gcd1

        fea_gcd1 = fea_gcd1.reshape(fea_gcd1.shape[0], fea_gcd1.shape[1], -1).transpose(1, 2)
        fea_gcd1 = self.gcd131(fea_gcd1, self.A11)
        fea_gcd1 = self.relu(fea_gcd1)
        fea_gcd1 = self.gcd132(fea_gcd1, self.A11)
        fea_gcd1 = self.relu(fea_gcd1)
        fea_gcd1 = fea_gcd1.transpose(1, 2).reshape(fea_gcdo13.shape)


        fea_gc1 = fea_gcd+fea_gcu
        fea_gc1 = self.convgc(fea_gc1)
        fea_gc2 = fea_gcd1+fea_gcu1
        fea_gc2 = self.convgc1(fea_gc2)
        fea_gc = fea_gc2 + fea_gc1
        fea_gc = self.convf(fea_gc)
        fea = fea_gc + f3
        low_level_feat3 = self.features.convres(low_level_feat3)
        fea = F.interpolate(fea, low_level_feat3.shape[2:] , mode='bilinear', align_corners=True)

        feao = torch.cat([fea, low_level_feat3], dim=1)
        out = self.features.decoder(feao)

        return out










