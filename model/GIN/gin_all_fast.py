# adopted from 'GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training, cai et al.'
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from model.GIN.readout import SumPooling, AvgPooling, MaxPooling


class ApplyNodeFunc(nn.Module):

    def __init__(self, mlp):#, norm_type
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        #self.norm = Norm(norm_type, self.mlp.output_dim)
        
        self.norm = nn.BatchNorm1d((self.mlp.output_dim))


    def forward(self, graph, h):
        h = self.mlp(graph, h)
        h = self.norm(h)
        h = F.relu(h)
        return h

class MLP(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        # self.GN = GN()

        if num_layers < 1:
            raise ValueError("number of layers should be postive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            self.norm_list = torch.nn.ModuleList()

            for layer in range(num_layers - 1):
                self.norm_list.append(nn.BatchNorm1d((hidden_dim)))


    def forward(self, graph, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = self.linears[i](h)
                h = self.norm_list[i](h)
                h = F.relu(h)

            return self.linears[-1](h)

class GINConv(nn.Module):
    def __init__(self, apply_func, aggregator_type, init_eps=0, learn_eps=False, hidden=64):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))

        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, g, split_list, feat):
        graph = g.local_var()
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_u('h', 'm'),
                         self._reducer('m', 'neigh'))
        rst = (1 + self.eps) * feat + graph.ndata['neigh']

        if self.apply_func is not None:
            rst = self.apply_func(g, rst)
        return rst

class GIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.ginlayers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps)
            )

        self.linears_prediction = nn.Linear(hidden_dim, output_dim)

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        hidden_rep = [h]
        split_list = g.batch_num_nodes

        for i in range(self.num_layers - 1):
            x = h
            h = self.ginlayers[i](g, split_list, h)

            if i != 0:
                h += x
            hidden_rep.append(h)

        score_over_layer = 0
        pooled_h = self.pool(g, hidden_rep[-1])
        score_over_layer += self.drop(self.linears_prediction(pooled_h))

        return score_over_layer
    
def GIN5_64(dataset):
    return GIN(num_layers=5, num_mlp_layers=2, input_dim=dataset.dim_nfeats, hidden_dim=64,
               output_dim=dataset.gclasses, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum')
    
def GIN5_32(dataset):
    return GIN(num_layers=5, num_mlp_layers=2, input_dim=dataset.dim_nfeats, hidden_dim=32,
               output_dim=dataset.gclasses, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum')

def GIN3_64(dataset):
    return GIN(num_layers=3, num_mlp_layers=2, input_dim=dataset.dim_nfeats, hidden_dim=64,
               output_dim=dataset.gclasses, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum')
def GIN3_32(dataset):
    return GIN(num_layers=3, num_mlp_layers=2, input_dim=dataset.dim_nfeats, hidden_dim=32,
               output_dim=dataset.gclasses, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum')

def GIN2_64(dataset):
    return GIN(num_layers=2, num_mlp_layers=2, input_dim=dataset.dim_nfeats, hidden_dim=64,
               output_dim=dataset.gclasses, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum')

def GIN2_32(dataset):
    return GIN(num_layers=2, num_mlp_layers=2, input_dim=dataset.dim_nfeats, hidden_dim=32,
               output_dim=dataset.gclasses, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum')