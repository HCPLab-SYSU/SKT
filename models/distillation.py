import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity(stu_map, tea_map):
    similiar = 1-F.cosine_similarity(stu_map, tea_map, dim=1)
    loss = similiar.sum()
    return loss


def cal_dense_fsp(features):
    fsp = []
    for groups in features:
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                x = groups[i]
                y = groups[j]

                norm1 = nn.InstanceNorm2d(x.shape[1])
                norm2 = nn.InstanceNorm2d(y.shape[1])
                x = norm1(x)
                y = norm2(y)
                res = gram(x, y)
                fsp.append(res)
    return fsp


def gram(x, y):
    n = x.shape[0]
    c1 = x.shape[1]
    c2 = y.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    x = x.view(n, c1, -1, 1)[0, :, :, 0]
    y = y.view(n, c2, -1, 1)[0, :, :, 0]
    y = y.transpose(0, 1)
    # print x.shape
    # print y.shape
    z = torch.mm(x, y) / (w*h)
    return z


def scale_process(features, scale=[3, 2, 1], ceil_mode=True):
    # process features for multi-scale dense fsp
    new_features = []
    for i in range(len(features)):
        if i >= len(scale):
            new_features.append(features[i])
            continue
        down_ratio = pow(2, scale[i])
        pool = nn.MaxPool2d(kernel_size=down_ratio, stride=down_ratio, ceil_mode=ceil_mode)
        new_features.append(pool(features[i]))
    return new_features