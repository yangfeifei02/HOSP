from torchtools import *

from collections import OrderedDict
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvBlock(nn.Module):       
    def __init__(self, in_planes, out_planes, userelu=True, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))

        if tt.arg.normtype == 'batch':
            self.layers.add_module('Norm', nn.BatchNorm2d(out_planes, momentum=momentum, affine=affine, track_running_stats=track_running_stats))
        elif tt.arg.normtype == 'instance':
            self.layers.add_module('Norm', nn.InstanceNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, opt, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvNet, self).__init__()
        self.in_planes  = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0),-1)
        return out

# encoder for imagenet dataset
class EmbeddingImagenet(nn.Module):
    def __init__(self,
                 emb_size):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                              out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))
        # 32*32
        # self.layer_last = nn.Sequential(nn.Linear(in_features=self.hidden * 4,
        #                                           out_features=self.emb_size, bias=True),
        #                                 nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        output_data = self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))
        return self.layer_last(output_data.view(output_data.size(0), -1))


class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,  #in_features = 128
                 num_features,    #num_features = 96
                 ratio=[3, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features    # 128
        self.num_features_list = [256,128]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else 512,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        # get eye matrix (batch_size x 2 x node_size x node_size)
        #diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).to(tt.arg.device)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 3, 1, 1).to(tt.arg.device)  
        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1) 

        # compute attention and aggregate
        # tmp= torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1)  #size = 80*30*10
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)     
        #node_feat size = 80*512*10
        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)   
        
        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)  
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,in_features,num_features,   # num_features = 96  in_features = 96
                 ratio=[2, 2, 1, 1],
                 ra=[1/2,1/4,1/8,1/16],
                 separate_dissimilarity=False,
                 features=True,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = 128
        #self.in_features = in_features
        #self.num_features_list = [num_features * r for r in ratio]    #  num_features_list [192,192,96,96]
        self.num_features_list = [self.in_features * r for r in ratio]
        self.num_feature_list = [self.in_features * r for r in ra]

        self.separate_dissimilarity = separate_dissimilarity    #False
        self.features = features

        self.dropout = dropout   # 0.1
        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],)
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)

        if self.features:
            layer_list = OrderedDict()
            for l in range(len(self.num_feature_list)):
                layer_list['linear{}'.format(l)] = nn.Linear(in_features=int(self.num_feature_list[l-1]) if l > 0 else self.in_features, out_features=int(self.num_feature_list[l]))
                layer_list['relu{}'.format(l)] = nn.ReLU()
            layer_list['linear_out'] = nn.Linear(in_features=int(self.num_feature_list[-1]),out_features=1)
        self.feat_sim = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = F.sigmoid(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val
          
       # compute high-order structure

        node = node_feat
        node = torch.abs(node)

        # for i in range(node.size(0)):
        #     for j in range(node.size(1)):
        #         node[i-1][j-1] = node[i][j] - node[i - 1][j - 1]
        #         j = j + 1
        # i = i + 1

        nod = torch.cat([node[1:, :, :], node[:1, :, :]], 0)
        nod = torch.cat([nod[:, 1:, :], nod[:, :1, :]], 1)
        node = node - nod
        node = node.cuda(device="cuda:3")
        x1 = node.unsqueeze(2)
        x2 = torch.transpose(x_i, 1, 2)
        node = torch.abs(x1 - x2)

        node = torch.abs(node)

        feat_sim = F.softmax(self.feat_sim(node))

        feat_sim = torch.transpose(feat_sim, 1, 3)

        mainfold = sim_val * edge_feat[:, 0, :, :] + (1.0 - sim_val) * edge_feat[:, 1, :, :] + feat_sim * edge_feat[:, 2, :, :]
        mainfold = torch.norm(mainfold, p='fro', dim=None, keepdim=False, out=None, dtype=None)

        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 3, 1, 1).to(
            tt.arg.device)
        # diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).to(tt.arg.device)
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val, feat_sim], 1) * edge_feat, p=1, dim=-1) * merge_sum
      
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0),
                                     torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0),
                                     torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(
            0).repeat(node_feat.size(0), 1, 1, 1).to(tt.arg.device)
             edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6  # size = 80*3*10*10
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 3, 1, 1)

        return edge_feat, mainfold


class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat,mainfold):
        # for each layer
        edge_feat_list = []
        mainfold_list = []
        for l in range(self.num_layers):
            # (1) edge to node
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)    # size = 80*10*128

            # (2) node to edge
            edge_feat,mainfold = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)
         
            # save edge feature
            edge_feat_list.append(edge_feat)
            mainfold_list.append(mainfold)  

        if tt.arg.visualization:
            for l in range(self.num_layers):
                ax = sns.heatmap(tt.nvar(edge_feat_list[l][0, 0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.1,  cmap="coolwarm",  cbar=False, square=True)
                ax.get_figure().savefig('./visualization/edge_feat_layer{}.png'.format(l))

        return edge_feat_list,mainfold_list
