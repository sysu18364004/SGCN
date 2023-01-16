import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

# import torch.nn.functional as F


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)
class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)
# class GraphConvolution(Module):

#     def __init__(self, in_features, out_features,  feature_less=None,drop_out = 0, activation=None, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.zeros(1, out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters(in_features, out_features)
#         self.dropout = torch.nn.Dropout(drop_out)
#         self.activation =  activation
#         if feature_less:
#             self.weight_less = Parameter(torch.FloatTensor(in_features, out_features))

#     def reset_parameters(self,in_features, out_features):
#         stdv = np.sqrt(6.0/(in_features+out_features))
#         # stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         # if self.bias is not None:
#         #     torch.nn.init.zeros_(self.bias)
#             # self.bias.data.uniform_(-stdv, stdv)


#     def forward(self, input, adj, feature_less = False):
#         if feature_less:           
#             support1 = torch.mm(input, self.weight)
#             support2 = self.weight
#             support = torch.vstack((support1,support2))
#             support = self.dropout(support)
#         else:    
#             input = self.dropout(input)
#             support = torch.spmm(input, self.weight)
#         output = torch.spmm(adj, support)
#         if self.bias is not None:
#             output = output + self.bias
#         if self.activation is not None:
#             output = self.activation(output)
#         return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid, dropout, activation = nn.ReLU())
#         self.gc2 = GraphConvolution(nhid, nclass*3, dropout)
#         # self.sigmoid = nn.Sigmoid()
#         self.fc = nn.Linear(nclass*3,nclass)
#         # self.fc = MLP(nclass*3,nclass,nclass,3,0.1)
#     def encode(self, x, adj):
#         x1 = self.gc1(x, adj, feature_less = True) 
#         x2 = self.gc2(x1,adj)
#         return x1,x2
#     def forward(self, x, adj):
#         x1 = self.gc1(x, adj, feature_less = True) 
#         x2 = self.gc2(x1,adj)
        
#         # print(x2.shape)
#         x2 = nn.functional.relu(x2)
#         x2 = self.fc(x2)
        
#         x2 = torch.sigmoid(x2)
#         # print(x2.shape)
#         return x1, x2               
import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
class GraphConvolution(Module):

    def __init__(self, in_features, out_features,  feature_less=None,drop_out = 0, activation=None, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(in_features, out_features)
        self.dropout = torch.nn.Dropout(drop_out)
        self.activation =  activation
        if feature_less:
            self.weight_less = Parameter(torch.FloatTensor(in_features, out_features))

    def reset_parameters(self,in_features, out_features):
        stdv = np.sqrt(6.0/(in_features+out_features))
        # stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     torch.nn.init.zeros_(self.bias)
            # self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj, feature_less = False):
        # print(input.shape)
        if feature_less:           
            # support1 = torch.mm(input, self.weight[:input.shape[1]])
            support1 = input
            support2 = self.weight
            support = torch.vstack((support1,support2))
            support = self.dropout(support)
        else:    
            input = self.dropout(input)
            support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
def fix(t):
    t[t < 0.] = 0.
    t[t > 1.] = 1.
def normal_vec(t):
    # norm = (t - t.mean(1).unsqueeze(1))/t.std(1).unsqueeze(1)
    # norm[torch.isnan(norm)] = 0.0
    tmo = 1/(torch.sqrt((t**2).sum(1))).unsqueeze(1)
    tmo[torch.isnan(tmo)] = 0.0
    return t * tmo
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, dropout, activation = nn.ReLU())
        self.gc2 = GraphConvolution(nhid, nclass*3, dropout)
        # self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(nclass*3,nclass)
        # self.fc = MLP(nclass*3,128, nclass,2,0.1)
        self.nclass = nclass
        # self.
        # self.fc = MLP(nhid,nclass,nclass,3,0.1)
        # self.BN = nn.BatchNorm1d(32)
    def encode(self, x, adj):

        # BN = nn.BatchNorm1d(adj.shape[0]).to(x.device)
        x1 = self.gc1(x, adj, feature_less = True) 
        x2 = self.gc2(x1,adj)
        x2 = nn.functional.relu(x2)
        x2 = self.fc(x2)
        # x2 = normal_vec(x2)

        # x2 = self.BN(x2)
        return x1,x2[-self.nclass:]
    def forward(self, x, adj):
        # print(adj.shape)
        x1 = self.gc1(x, adj, feature_less = True) 
        x2 = self.gc2(x1,adj)
        
        # print(x2.shape)
        x2 = nn.functional.relu(x2)
        x2 = self.fc(x2)
        x2 = torch.sigmoid(x2)
        return x1, x2 
        # x2 = torch.spmm(label_adj,x2.T).T
        # x2 = torch.sigmoid(x2)
        # x2 = normal_vec(x2)

        # label_embedding = x2[-self.nclass:]
        # cls1 = torch.mm(x2,label_embedding.T)
        # # print(cls1)
        # x2 = torch.sigmoid(cls1)
        # # print(x2.shape)
        # return x1, x2      
#         # print(adj)
#         # BN = nn.BatchNorm1d(adj.shape[0]).to(x.device)
#         x1 = self.gc1(x, adj, feature_less = True) 
#         x2 = self.gc2(x1,adj)
#         # x2 = nn.functional.relu(x2)
#         fix(x2)
#         x2 = self.fc(x2)
#         # x2 = normal_vec(x2)
#         # print(x2)
#         # x2 = BN(x2.unsqueeze(0))[0]
#         # x2 = self.BN(x2)
        
#         # label_embedding = x2[-self.nclass:]

        
#         # cls1 = torch.mm(x2,label_embedding.T)
#         # fix(cls1)
# #         print(cls)
#         # print(cls1.min(),cls1.max())
#         # cls1 = torch.sigmoid(cls1)
#         # print(x2.shape)
#         # print((cls1>0).sum())
#         return x1, cls1               