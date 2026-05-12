"""
BWGNN 模型定义 — 基于 Beta 小波的图神经网络
改编自: Rethinking Graph Neural Networks for Anomaly Detection (ICML 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy
import numpy as np
from torch.nn import init


def calculate_theta2(d):
    """计算 Beta 小波的多项式系数 (d+1 个基)"""
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly(
            (x / 2) ** i * (1 - x / 2) ** (d - i)
            / scipy.special.beta(i + 1, d + 1 - i)
        )
        coeff = f.all_coeffs()
        inv_coeff = [float(coeff[d - j]) for j in range(d + 1)]
        thetas.append(inv_coeff)
    return thetas


class PolyConv(nn.Module):
    """多项式图卷积层: 使用 Chebyshev 多项式近似图滤波器"""

    def __init__(self, in_feats, out_feats, theta, activation=F.leaky_relu, lin=False, bias=False):
        super().__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(
                graph.in_degrees().float().clamp(min=1), -0.5
            ).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


class BWGNN(nn.Module):
    """
    Beta Wavelet GNN — 同构图异常检测模型

    核心思想: 利用 Beta 小波基在图谱空间上的多尺度分析能力，
    捕获异常节点在不同频率尺度上的差异特征。
    """

    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList([
            PolyConv(h_feats, h_feats, theta, lin=False)
            for theta in self.thetas
        ])
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, in_feat):
        h = self.act(self.linear(in_feat))
        h = self.act(self.linear2(h))
        h_final = torch.zeros([len(in_feat), 0], device=in_feat.device)
        for conv in self.conv:
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
        h = self.act(self.linear3(h_final))
        h = self.linear4(h)
        return h

    def get_embeddings(self, in_feat):
        """提取节点嵌入向量 (用于下游风险洞察分析)"""
        h = self.act(self.linear(in_feat))
        h = self.act(self.linear2(h))
        h_final = torch.zeros([len(in_feat), 0], device=in_feat.device)
        for conv in self.conv:
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
        embeddings = self.act(self.linear3(h_final))
        return embeddings


class BWGNN_Hetero(nn.Module):
    """Beta Wavelet GNN — 异构图版本"""

    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        self.conv = nn.ModuleList([
            PolyConv(h_feats, h_feats, theta, lin=False)
            for theta in self.thetas
        ])
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()

    def forward(self, in_feat):
        h = self.act(self.linear(in_feat))
        h = self.act(self.linear2(h))
        h_all = []
        for relation in self.g.canonical_etypes:
            h_final = torch.zeros([len(in_feat), 0], device=in_feat.device)
            for conv in self.conv:
                h0 = conv(self.g[relation], h)
                h_final = torch.cat([h_final, h0], -1)
            h_r = self.linear3(h_final)
            h_all.append(h_r)
        h_all = torch.stack(h_all).sum(0)
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        return h_all
