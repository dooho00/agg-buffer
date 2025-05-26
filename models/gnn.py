import dgl
import torch
import math
import torch.nn as nn
import torch.nn.functional as F 

class GCN(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout=0.0,
                 use_linear=False,
                 norm='identity',
                 prelu=False,
                 encoder_mode=False,
                 mp_norm='both'):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.encoder_mode = encoder_mode
        self.hidden_lst = [in_feats] + hidden_lst
        self.use_linear = use_linear

        if norm == 'layer':
            norm = torch.nn.LayerNorm
        elif norm == 'batch':
            norm = torch.nn.BatchNorm1d
        else: 
            norm = torch.nn.Identity

        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            self.layers.append(dgl.nn.GraphConv(in_, out_, allow_zero_in_degree=True, norm=mp_norm))
            self.norms.append(norm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]

    def forward(self, g, features, return_all=False):
        h = features
        stack = []
        for i, layer in enumerate(self.layers):
            # dropout
            if i != 0: h = self.dropout(h)

            # apply linear
            if self.use_linear:
                linear = self.linears[i](h)
            # graph conv
            h = layer(g, h)

            # residual connection
            if self.use_linear:
                h = h + linear

            # activation and norm
            if i != len(self.layers) - 1 or self.encoder_mode:
                h = self.activations[i](self.norms[i](h))
            stack.append(h)
            
        return stack if return_all else stack[-1]
    
class MLP(nn.Module):
    def __init__(
                self,
                in_feats,
                hidden_lst,
                dropout=0.0,
                use_linear=False,
                norm='identity',
                prelu=False,
                encoder_mode=False,
                mp_norm='both'):

        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.encoder_mode = encoder_mode
        self.hidden_lst = [in_feats] + hidden_lst

        if norm == 'layer':
            norm = torch.nn.LayerNorm
        elif norm == 'batch':
            norm = torch.nn.BatchNorm1d
        else: 
            norm = torch.nn.Identity

        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            self.layers.append(nn.Linear(in_, out_))
            self.norms.append(norm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
        
        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]

    def forward(self, features, return_all=False):
        h = features
        stack = []
        for i, layer in enumerate(self.layers):
            # dropout
            if i != 0: h = self.dropout(h)
            h = layer(h)

            # activation and norm
            if i != len(self.layers) - 1 or self.encoder_mode:
                h = self.activations[i](self.norms[i](h))

        stack.append(h)

        return stack if return_all else stack[-1]
    
class SAGE(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout=0.5,
                 use_linear=False,
                 norm='identity',
                 prelu=False,
                 encoder_mode=False,
                 mp_norm='both'):
        super(SAGE, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.encoder_mode = encoder_mode
        self.hidden_lst = [in_feats] + hidden_lst
        self.use_linear = use_linear

        if norm == 'layer':
            norm = torch.nn.LayerNorm
        elif norm == 'batch':
            norm = torch.nn.BatchNorm1d
        else: 
            norm = torch.nn.Identity

        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            self.layers.append(dgl.nn.SAGEConv(in_, out_, aggregator_type = "mean"))
            self.norms.append(norm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]

    def forward(self, g, features, return_all=False):
        h = features
        stack = []
        for i, layer in enumerate(self.layers):
            # dropout
            if i != 0: h = self.dropout(h)

            # apply lnear
            if self.use_linear:
                linear = self.linears[i](h)
            # graph conv
            h = layer(g, h)

            # res
            if self.use_linear:
                h = h + linear

            # activation and norm
            if i != len(self.layers) - 1 or self.encoder_mode:
                h = self.activations[i](self.norms[i](h))
            stack.append(h)
            
        return stack if return_all else stack[-1]

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, use_linear=False, dropout=0.0, heads=[8,1], norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GAT
        self.dropout = dropout
        self.hidden_lst = [in_size, hid_size]
        self.use_linear = use_linear

        if norm == 'layer':
            norm = torch.nn.LayerNorm
        elif norm == 'batch':
            norm = torch.nn.BatchNorm1d
        else: 
            norm = torch.nn.Identity

        self.linears = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for in_, out_ in zip([in_size, hid_size], [hid_size, out_size]):
            self.norms.append(norm(out_))
            self.activations.append(torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))

        self.layers.append(
            dgl.nn.GATConv(
                in_size,
                hid_size // heads [0],
                heads[0],
                feat_drop=self.dropout,
                attn_drop=0.3,
                activation=None,
                allow_zero_in_degree=True
            )
        )
        self.layers.append(
            dgl.nn.GATConv(
                hid_size,
                out_size,
                heads[1],
                feat_drop=self.dropout,
                attn_drop=0.3,
                activation=None,
                allow_zero_in_degree=True
            )
        )

    def forward(self, g, inputs, return_all=False):
        h = inputs
        for i, layer in enumerate(self.layers):
            h_conv = layer(g, h)

            if i == 1:  # last layer
                h_conv = h_conv.mean(1)
            else:  # other layer(s)
                h_conv = h_conv.flatten(1)

            if self.use_linear:
                h_conv = h_conv + self.linears[i](h)
            
            # activation and norm
            if i != len(self.layers) - 1:
                h = self.activations[i](self.norms[i](h_conv))
            else:
                h = h_conv
        return h

class GIN(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout=0.5,
                 use_linear=False,
                 norm='identity',
                 prelu=False,
                 encoder_mode=False,
                 mp_norm='both'):
        super(GIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.encoder_mode = encoder_mode
        self.hidden_lst = [in_feats] + hidden_lst
        self.use_linear = use_linear

        if norm == 'layer':
            norm = torch.nn.LayerNorm
        elif norm == 'batch':
            norm = torch.nn.BatchNorm1d
        else: 
            norm = torch.nn.Identity

        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            self.layers.append(dgl.nn.GINConv(torch.nn.Linear(in_, out_)))
            self.norms.append(norm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]

    def forward(self, g, features, return_all=False):
        h = features
        stack = []
        for i, layer in enumerate(self.layers):
            # dropout
            if i != 0: h = self.dropout(h)

            # apply linear
            if self.use_linear:
                linear = self.linears[i](h)
            # graph conv
            h = layer(g, h)

            # res
            if self.use_linear:
                h = h + linear

            # activation and norm
            if i != len(self.layers) - 1 or self.encoder_mode:
                h = self.activations[i](self.norms[i](h))
            stack.append(h)
            
        return stack if return_all else stack[-1]

class SGC(torch.nn.Module):
    def __init__(self,
                in_feats,
                hidden_lst,
                dropout=0.0,
                use_linear=False,
                norm='identity',
                prelu=False,
                encoder_mode=False,
                mp_norm='both'):
        super(SGC, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.encoder_mode = encoder_mode
        self.hidden_lst = [in_feats] + hidden_lst
        self.layers.append(torch.nn.Linear(in_feats, hidden_lst[-1], bias = True))
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, g, feat, return_all=False):
        degs = g.in_degrees().float().clamp(min=1).to(feat.device).unsqueeze(1)
        norm = torch.pow(degs, -0.5)

        feat = self.layers[0](feat)

        for i in range(len(self.hidden_lst)-1):
            # Normalize features
            feat = feat * norm
            g.ndata['h'] = feat
            # Message passing
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
            feat = g.ndata.pop('h') * norm        
        
        return feat