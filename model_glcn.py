import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
from layers import GraphConvolution
import numpy as np

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']

__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def pair_wise_distance(inputs):
    n = inputs.size(0)
    d = inputs.size(1)
    S_i = inputs.unsqueeze(1).expand(n, n, d)
    S_j = inputs.unsqueeze(0).expand(n, n, d)

    dist = torch.pow(S_i - S_j, 2)

    return dist


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''

    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def pairwise_distances_element(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''

    # make x -> (d, N, 1)
    x = torch.transpose(x, 0 ,1)
    x = x.unsqueeze(-1)
    x_square = x ** 2

    if y is not None:
        # make x -> (d, 1, N)
        y = torch.transpose(y, 0, 1)
        y = y.unsqueeze(1)
        y_square = y ** 2
    else:
        # make x -> (d, 1, N)
        y = torch.transpose(x, 1, 2)
        y_square = torch.transpose(x_square, 1, 2)

    dist = x_square + y_square - 2.0 * torch.matmul(x, y)

    # make the dist matrix to be NxMxd
    dist = dist.permute(1, 2, 0)

    torch.clamp(dist, 0.0, np.inf)
    dist[dist != dist] = 0

    return dist



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)


class GraphLearning(nn.Module):
    def __init__(self, in_channels, out_channels, top_bn=True,
                 batch_size=100, total_num=10000, topk=50):

        super(GraphLearning, self).__init__()
        self.top_bn = top_bn
        self.batch_size = batch_size
        self.total_num = total_num
        self.topk = topk

        # self.main = nn.Sequential(
        #     nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.1),
        #
        #     nn.Conv2d(32, 32, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.1),
        #
        #     #
        #     # nn.Conv2d(128, 128, 3, 1, 1, bias=False),
        #     # nn.BatchNorm2d(128),
        #     # nn.LeakyReLU(0.1),
        #     #
        #     # nn.MaxPool2d(2, 2, 1),
        #     # nn.Dropout2d(),
        #     #
        #     # nn.Conv2d(128, 256, 3, 1, 1, bias=False),
        #     # nn.BatchNorm2d(256),
        #     # nn.LeakyReLU(0.1),
        #     #
        #     # nn.Conv2d(256, 256, 3, 1, 1, bias=False),
        #     # nn.BatchNorm2d(256),
        #     # nn.LeakyReLU(0.1),
        #     #
        #     # nn.Conv2d(256, 256, 3, 1, 1, bias=False),
        #     # nn.BatchNorm2d(256),
        #     # nn.LeakyReLU(0.1),
        #     #
        #     # nn.MaxPool2d(2, 2, 1),
        #     # nn.Dropout2d(),
        #     #
        #     # nn.Conv2d(256, 512, 3, 1, 0, bias=False),
        #     # nn.BatchNorm2d(512),
        #     # nn.LeakyReLU(0.1),
        #     #
        #     # nn.Conv2d(512, 256, 1, 1, 1, bias=False),
        #     # nn.BatchNorm2d(256),
        #     # nn.LeakyReLU(0.1),
        #     #
        #     # nn.Conv2d(256, 128, 1, 1, 1, bias=False),
        #     # nn.BatchNorm2d(128),
        #     # nn.LeakyReLU(0.1),
        #
        #     # nn.AdaptiveAvgPool2d((1, 1))
        # )

        self.linear = nn.Linear(32*32*in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        # linear transform to 1
        self.S_linear = nn.Linear(out_channels, 1)

    def forward(self, inputs):

        outputs = []
        for start_idx in range(0, self.total_num, self.batch_size):
            # print(start_idx)
            output = self.main(inputs[start_idx:start_idx+self.batch_size])
            output = self.linear(output.view(self.batch_size, -1))
            if self.top_bn:
                output = self.bn(output)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        # print("outputs shape is ", outputs.shape)
        S = self.construct_graph_S_loop(outputs)


        return outputs, S

    def construct_graph_S(self, inputs):
        """
        S_{i, j} = \frac{exp(ReLU(a^{T}|x_i - x_j|))}{\sum_{j=1}{n}(ReLU(a^{T}|x_i - x_j|))}

        :param inputs: X \in R^{n \times d}
        :return: S
        """

        dist = pairwise_distances_element(inputs)
        dist = F.relu(self.S_linear(dist)).squeeze()
        # print("dist shape after a^{t} |x_i - x_j|", dist.shape)
        dist = torch.exp(dist)
        dist = dist / torch.sum(dist, dim=-1, keepdim=True)
        print(dist.max())

        return dist

    def construct_graph_S_loop(self, inputs):
        """
        S_{i, j} = \frac{exp(ReLU(a^{T}|x_i - x_j|))}{\sum_{j=1}{n}(ReLU(a^{T}|x_i - x_j|))}

        :param inputs: X \in R^{n \times d}
        :return: S
        """

        dist = self.pairwise_distances_func_loop(inputs)

        return dist

    def pairwise_distances_func_loop(self, x):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''

        N = x.size(0)
        S = torch.zeros((N, N), device=x.device)
        row_indices = []
        col_indices = []
        values = []
        for i in range(N):
            x_i = x[i].view(1, -1)
            # dist_i shape is (1, N, d)
            dist_i = pairwise_distances_element(x_i, x)
            # non_neg_dist_i  shape is (1, N)
            non_neg_dist_i = F.relu(self.S_linear(dist_i)).squeeze()
            tmp_values, tmp_indices = torch.topk(non_neg_dist_i, self.topk)
            tmp_values = F.softmax(tmp_values)
            S[i, tmp_indices] = tmp_values

            # row_indices += [i] * self.topk
            # col_indices.append(tmp_indices)
            # tmp_values = F.softmax(tmp_values)
            # values.append(tmp_values)

        # row_indices = torch.tensor(row_indices).long().to(x.device)
        # col_indices = torch.cat(col_indices)
        # values = torch.cat(values)
        # indices = torch.stack([row_indices, col_indices], dim=0)
        # S = torch.sparse_coo_tensor(indices, values, (N, N), device=x.device)
            # # print(i, dist_i.shape)
            # S[i] = F.relu(self.S_linear(dist_i)).squeeze()
        # S = torch.softmax(S, dim=-1)

        return S


class GLCN(nn.Module):
    def __init__(self, in_channels=3, out_channels=7, ngcn_layers=30,
                 nclass=10, gamma_reg=0.01, dropout=0.2, topk=50):
        super(GLCN, self).__init__()

        self.gamma_reg = gamma_reg
        self.graph_learning = GraphLearning(in_channels, out_channels, top_bn=True, topk=topk)
        self.gcn = GCN(out_channels, ngcn_layers, nclass, dropout)

    @staticmethod
    def sparse_dense_mul(s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sum(dv * v)

    def forward(self, inputs):

        extract_feature, S = self.graph_learning(inputs)
        loss_GL = pairwise_distances(extract_feature)

        # S_dense = S.to_dense()
        # loss_GL = torch.sum(feature_dist, dim=-1)
        loss_GL = torch.sum(loss_GL * S) + self.gamma_reg * torch.sum(S)
        # print("Extracted Feature is ", extract_feature.shape)
        semi_outputs = self.gcn(extract_feature, S)


        return semi_outputs, loss_GL, S











