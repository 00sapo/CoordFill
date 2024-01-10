import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import pretrained_networks as pn


def normalize_tensor(in_feat, eps=1e-10):
    # norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3]).repeat(1,in_feat.size()[1],1,1)
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1)).view(
        in_feat.size()[0], 1, in_feat.size()[2], in_feat.size()[3]
    )
    return in_feat / (norm_factor.expand_as(in_feat) + eps)


def cos_sim(in0, in1):
    in0_norm = normalize_tensor(in0)
    in1_norm = normalize_tensor(in1)
    N = in0.size()[0]
    X = in0.size()[2]
    Y = in0.size()[3]

    return torch.mean(
        torch.mean(torch.sum(in0_norm * in1_norm, dim=1).view(N, 1, X, Y), dim=2).view(
            N, 1, 1, Y
        ),
        dim=3,
    ).view(N)


def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1, 2, 0))


def tensor2im(image_tensor, imtype=np.uint8, cent=1.0, factor=255.0 / 2.0):
    # def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)


# Off-the-shelf deep network
class PNet(nn.Module):
    """Pre-trained network with all channels equally weighted by default"""

    def __init__(self, pnet_type="vgg", pnet_rand=False):
        super(PNet, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_rand = pnet_rand

        self.shift = torch.autograd.Variable(
            torch.Tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1)
        )
        self.scale = torch.autograd.Variable(
            torch.Tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1)
        )

        if self.pnet_type in ["vgg", "vgg16"]:
            self.net = pn.vgg16(pretrained=not self.pnet_rand, requires_grad=False)
        elif self.pnet_type == "alex":
            self.net = pn.alexnet(pretrained=not self.pnet_rand, requires_grad=False)
        elif self.pnet_type[:-2] == "resnet":
            self.net = pn.resnet(
                pretrained=not self.pnet_rand,
                requires_grad=False,
                num=int(self.pnet_type[-2:]),
            )
        elif self.pnet_type == "squeeze":
            self.net = pn.squeezenet(pretrained=not self.pnet_rand, requires_grad=False)

        self.L = self.net.N_slices

    def forward(self, in0, in1, retPerLayer=False):
        in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)

        outs0 = self.net.forward(in0_sc)
        outs1 = self.net.forward(in1_sc)

        if retPerLayer:
            all_scores = []
        for kk, out0 in enumerate(outs0):
            cur_score = 1.0 - cos_sim(outs0[kk], outs1[kk])
            if kk == 0:
                val = 1.0 * cur_score
            else:
                # val = val + self.lambda_feat_layers[kk]*cur_score
                val = val + cur_score
            if retPerLayer:
                all_scores += [cur_score]

        if retPerLayer:
            return (val, all_scores)
        else:
            return val


# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(
        self,
        pnet_type="vgg",
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        spatial=False,
        version="0.1",
    ):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.version = version

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]

        if self.pnet_tune:
            self.net = net_type(pretrained=not self.pnet_rand, requires_grad=True)
        else:
            self.net = [
                net_type(pretrained=not self.pnet_rand, requires_grad=True),
            ]

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == "squeeze":  # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lins += [self.lin5, self.lin6]

        self.shift = torch.autograd.Variable(
            torch.Tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1)
        )
        self.scale = torch.autograd.Variable(
            torch.Tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1)
        )

    def forward(self, in0, in1):
        in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)

        if self.version == "0.0":
            # v0.0 - original release had a bug, where input was not scaled
            in0_input = in0
            in1_input = in1
        else:
            # v0.1
            in0_input = in0_sc
            in1_input = in1_sc

        if self.pnet_tune:
            outs0 = self.net.forward(in0_input)
            outs1 = self.net.forward(in1_input)
        else:
            outs0 = self.net[0].forward(in0_input)
            outs1 = self.net[0].forward(in1_input)

        feats0 = {}
        feats1 = {}
        diffs = [0] * len(outs0)

        for kk, out0 in enumerate(outs0):
            feats0[kk] = normalize_tensor(outs0[kk])
            feats1[kk] = normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

            # diffs[kk] = (outs0[kk]-outs1[kk])**2

        if self.spatial:
            lin_models = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == "squeeze":
                lin_models.extend([self.lin5, self.lin6])
            res = [lin_models[kk].model(diffs[kk]) for kk in range(len(diffs))]
            return res

        val1 = torch.mean(torch.mean(self.lin0.model(diffs[0]), dim=3), dim=2)
        val2 = torch.mean(torch.mean(self.lin1.model(diffs[1]), dim=3), dim=2)
        val3 = torch.mean(torch.mean(self.lin2.model(diffs[2]), dim=3), dim=2)
        val4 = torch.mean(torch.mean(self.lin3.model(diffs[3]), dim=3), dim=2)
        val5 = torch.mean(torch.mean(self.lin4.model(diffs[4]), dim=3), dim=2)

        val = val1 + val2 + val3 + val4 + val5
        val_out = val.view(val.size()[0], val.size()[1], 1, 1)

        val_out2 = [val1, val2, val3, val4, val5]

        if self.pnet_type == "squeeze":
            val6 = val + torch.mean(torch.mean(self.lin5.model(diffs[5]), dim=3), dim=2)
            val7 = val6 + torch.mean(
                torch.mean(self.lin6.model(diffs[6]), dim=3), dim=2
            )

            val7 = val7.view(val7.size()[0], val7.size()[1], 1, 1)
            return val7

        return val_out, val_out2
        # return [val1, val2, val3, val4, val5]


class Dist2LogitLayer(nn.Module):
    """takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True)"""

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [
            nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),
        ]
        layers += [
            nn.LeakyReLU(0.2, True),
        ]
        layers += [
            nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),
        ]
        layers += [
            nn.LeakyReLU(0.2, True),
        ]
        layers += [
            nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),
        ]
        if use_sigmoid:
            layers += [
                nn.Sigmoid(),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(
            torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1)
        )


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()
        self.model = nn.Sequential(*[self.net])

    def forward(self, d0, d1, judge):
        per = (judge + 1.0) / 2.0
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, colorspace="Lab"):
        super(FakeNet, self).__init__()
        self.colorspace = colorspace


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("Network", net)
    print("Total number of parameters: %d" % num_params)
