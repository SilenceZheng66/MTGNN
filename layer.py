from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # Einstein Summation, 在最后两维做点积
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        # 返回一个与原始张量相同的张量，但是保证其内存布局是连续的
        # 这意味着如果原始张量内存上不连续则创建一个新的拷贝
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()  # 创建一个节点卷积层
        # (gdep + 1) * c_in 是 out拼接后第二维的大小
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep  # 图卷积的深度k，默认是2
        self.dropout = dropout  # 随机失活率
        self.alpha = alpha  # 超参数

    def forward(self, x, adj):
        # 将自连接加到邻接矩阵上
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # A+I
        # 纵向求和，计算节点的度，注意是浮点，并不是整数
        d = adj.sum(1)
        h = x  # 初始化节点特征H_in
        out = [h]  # 存储传播过程中的输出
        # d.view(-1, 1)把d竖起来，d.view(-1, 1)的横向扩展为D～
        # 然后用D～的逆右乘A+I，计算公式为adj / d.view(-1, 1)
        a = adj / d.view(-1, 1)  # 归一化邻接矩阵
        # 信息传播层⬇️，按图卷积的深度做了几轮运算
        for i in range(self.gdep):
            # H(k)的计算公式
            # self.alpha * x是初始节点保持因子，用于解决节点表征趋同
            # 但加入节点保持因子会削弱一些节点信息
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)  # 每次计算结果都添加上
        # 在第二维（通道）拼接out数组为一个张量
        ho = torch.cat(out, dim=1)
        # TODO：疑问1：为什么只有一个mlp
        # 猜想：out聚合起来变成ho，相当于多个mlp也聚合成一个mlp，相当于图上画的先mlp再加变成先加再mlp
        ho = self.mlp(ho)
        # TODO：疑问2：信息选择策略里的 W^k 参数矩阵在哪？
        # 猜想：目前看只能是在mlp层里？还得读一下原文
        return ho  # 返回最终的输出


class dy_mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        self.mlp2 = linear((gdep + 1) * c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)

    def forward(self, x):
        # adj = adj + torch.eye(adj.size(0)).to(x.device)
        # d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1 + ho2


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        # inception的卷积核，参照了自然时序周期
        self.kernel_set = [2, 3, 6, 7]
        # 根据卷积核数量平均分配输出通道数，确保每个卷积核的输出通道数相近
        cout = int(cout / len(self.kernel_set))
        # Inception的每个卷积部分都采用空洞卷积
        for kern in self.kernel_set:
            # 膨胀因子，用更少的层来更积极地合并整个输入的空间信息，感受域增长更快
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        # Inception forward
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        # 对每一个卷积核输出的特征在第四维上进行统一，这个不相等是由卷积核大小不同导致的
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]  # 统一为x[-1].size(3)，最后一个卷积核得到的特征图在第4维度上的大小，核大->特征小
        # 拼接各卷积核计算结果
        x = torch.cat(x, dim=1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        # lin1和lin2是可学习的参数
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha  # 激活函数的饱和率
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            # 节点嵌入到高维空间，获得初始节点嵌入
            # TODO：在节点嵌入加入更多信息，论文提到了
            nodevec1 = self.emb1(idx)  # 源节点嵌入
            nodevec2 = self.emb2(idx)  # 目标节点嵌入
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        # 计算邻接矩阵的非对称信息（认为节点变量间关系是单向的）
        # a通过两者相减表示单向关系
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        # Relu激活，正则化邻接矩阵
        adj = F.relu(torch.tanh(self.alpha * a))
        # 构造与邻接矩阵大小相同的 0 mask
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        # + torch.rand_like(adj) * 0.01 是 [0,1]的均匀分布缩放0.01，不知道为什么加这个
        # s1 是 最大值， t1 是 索引
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        # mask最终是adj上前k大元素置为1，其余置为0的矩阵
        mask.scatter_(1, t1, s1.fill_(1))
        # 对应元素相乘，只保留前k大值
        adj = adj * mask
        # 好像是在硬构造边
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
