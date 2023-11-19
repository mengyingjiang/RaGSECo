import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drug_encoding = 'CNN'


def posi_neg_gene(x_inter_fea):
    x_inter_fea = F.normalize(x_inter_fea)
    DDS = torch.mm(x_inter_fea, x_inter_fea.T)
    row, clomn = torch.where(DDS >= 0.95)
    a = row - clomn
    row = row[a >= 0]
    clomn = clomn[a >= 0]
    row = row[:200]
    clomn = clomn[:200]

    row_neg, clomn_neg = torch.where(DDS <= 0.1)
    b = row_neg - clomn_neg
    row_neg = row_neg[b >= 0]
    clomn_neg = clomn_neg[b >= 0]
    row_neg = row_neg[0:row.shape[0]]
    clomn_neg = clomn_neg[0:row.shape[0]]

    return row, clomn, row_neg, clomn_neg, clomn.shape[0], clomn_neg.shape[0]


class Discriminator_dd(nn.Module):
    def __init__(self, n_h):
        super(Discriminator_dd, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, view_1, view_2, row, clomn, row_neg, clomn_neg,n_posi, n_neg):
        if n_posi==0:
            sc_pos1=sc_pos2= torch.tensor([]).to(device)
        else:
            sc_pos1 = self.f_k(view_1[row, :], view_2[clomn, :])
            sc_pos2 = self.f_k(view_1[clomn, :], view_2[row, :])

        if n_neg==0:
            sc_neg1=sc_neg2 = torch.tensor([]).to(device)
        else:
            sc_neg1 = self.f_k(view_1[row_neg,:], view_2[clomn_neg,:])
            sc_neg2 = self.f_k(view_1[clomn_neg,:], view_2[row_neg,:])

        return torch.cat((sc_pos1, sc_pos2, sc_neg1, sc_neg2), 0)

class CNN_concat(nn.Sequential):
    def __init__(self, out_dim, encoding,  **config):
        super(CNN_concat, self).__init__()
        if encoding == 'drug':
            in_ch = [64] + config['cnn_drug_filters']
            kernels = config['cnn_drug_kernels']
            layer_size = len(config['cnn_drug_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                 out_channels=in_ch[i + 1],
                                                 kernel_size=kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((64, 200))
            self.fc1 = nn.Linear(n_size_d, out_dim)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v

class FNN(torch.nn.Module):  # Joining together
    def __init__(self, vector_size,dim,args):
        super(FNN, self).__init__()
        self.vector_size = vector_size

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + dim) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + dim) // 2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.l2 = torch.nn.Linear((self.vector_size + dim) // 2, dim)

        self.att2s = torch.nn.ModuleList(
           [EncoderLayer((self.vector_size + dim) // 2, args.bert_n_heads) for _ in range(1)])

        self.dr = torch.nn.Dropout(args.drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(F.relu(self.l1(X)))

        for att2 in self.att2s:
            X = att2(X)

        X = self.dr(F.relu(self.l2(X)))
        return X

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):  # [1966,4]
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)   # [1966,4]
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)  # 多头注意力去聚合其他DDI的特征
        X = self.AN1(output + X)  # 残差连接+LayerNorm

        output = self.l1(X)  # FC
        X = self.AN2(output + X)  # 残差连接+LayerNorm

        return X

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):   # [1966,4]

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads  # 491.5
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim  #1966
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        scores = torch.nn.Softmax(dim=-1)(scores)
        scores = torch.matmul(scores, V)
        # context: [len_q, n_heads * d_v]
        scores = scores.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        scores = self.fc(scores)
        return scores
class RGCN(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, N_relation, in_features, out_features, bias=True):
        super(RGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N_relation = N_relation
        self.weight = torch.nn.Parameter(torch.FloatTensor(N_relation, in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = torch.matmul(adj, input)
        output = torch.matmul(input, self.weight)
        output = torch.sum(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HomoGCN(torch.nn.Module):
    def __init__(self, in_features, out_features, n_hop,bias=True):
        super(HomoGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_hop = n_hop
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        for i in range(self.n_hop):
            input = torch.mm(adj, input)

        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class my_loss1(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()
        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.b_xent = nn.BCEWithLogitsLoss()
    def forward(self, X, target, ss , lbl):
        loss_ss= self.b_xent(ss, lbl.float())
        loss = 5 * self.criteria1(X, target.long()) + loss_ss
        return loss


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def generate_config(drug_encoding=None,
                    result_folder="./result/",
                    input_dim_drug=1024,
                    hidden_dim_drug=256,
                    cls_hidden_dims=[1024, 1024, 512],
                    mlp_hidden_dims_drug=[1024, 256, 64],
                    batch_size=256,
                    train_epoch=10,
                    test_every_X_epoch=20,
                    LR=1e-4,
                    mpnn_hidden_size=50,
                    mpnn_depth=3,
                    cnn_drug_filters=[32, 64, 96],
                    cnn_drug_kernels=[4, 6, 8],
                    num_workers=0,
                    cuda_id=None,
                    ):
    base_config = {'input_dim_drug': input_dim_drug,
                   'hidden_dim_drug': hidden_dim_drug,  # hidden dim of drug
                   'cls_hidden_dims': cls_hidden_dims,  # decoder classifier dim 1
                   'batch_size': batch_size,
                   'train_epoch': train_epoch,
                   'test_every_X_epoch': test_every_X_epoch,
                   'LR': LR,
                   'drug_encoding': drug_encoding,
                   'result_folder': result_folder,
                   'binary': False,
                   'num_workers': num_workers,
                   'cuda_id': cuda_id
                   }
    if not os.path.exists(base_config['result_folder']):
        os.makedirs(base_config['result_folder'])
    if drug_encoding == 'Morgan':
        base_config['mlp_hidden_dims_drug'] = mlp_hidden_dims_drug  # MLP classifier dim 1
    elif drug_encoding == 'CNN':
        base_config['cnn_drug_filters'] = cnn_drug_filters
        base_config['cnn_drug_kernels'] = cnn_drug_kernels
    # raise NotImplementedError
    elif drug_encoding is None:
        pass
    else:
        raise AttributeError("Please use the correct drug encoding available!")

    return base_config

config = generate_config(drug_encoding = drug_encoding,
                         cls_hidden_dims = [1024,1024,512],
                         train_epoch = 5,
                         LR = 0.001,
                         batch_size = 128,
                         hidden_dim_drug = 700,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3
                        )
