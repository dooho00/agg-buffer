import dgl
import torch
import math
import torch.nn as nn
import torch.nn.functional as F 
import dgl.function as fn
from dgl import DropEdge

class GCN_B(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout=0.5,
                 use_linear=False,
                 norm='identity',
                 prelu=False,
                 encoder_mode=False,
                 mp_norm='both'):
        super(GCN_B, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms_buffer = torch.nn.ModuleList()
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

        # parameters of aggregation buffer
        self.layer_buffers = torch.nn.ModuleList()
        
        accumulated_dim = in_feats  # Initialize with input features dimension
        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            self.layers.append(dgl.nn.GraphConv(in_, out_, allow_zero_in_degree=True, norm=mp_norm))
            self.norms_buffer.append(norm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))
            
            # Initialize buffer in each layer
            layer_buffer = torch.nn.Linear(accumulated_dim, out_, bias=True)
            torch.nn.init.constant_(layer_buffer.weight, 0.0)
            if layer_buffer.bias is not None:
                torch.nn.init.constant_(layer_buffer.bias, 0.0)
            self.layer_buffers.append(layer_buffer)
            
            # Accumulate dimensions for the next layer
            accumulated_dim += out_

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]
        
    def forward(self, g, features, return_all=False):
        h = features
        stack = [features]

        # Prepare normalization for layer buffer
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -1).unsqueeze(-1) 

        for i, layer in enumerate(self.layers):
            # dropout
            if i != 0: h = self.dropout(h)

            # apply linear
            if self.use_linear:
                linear = self.linears[i](h)
            # graph conv
            h_conv = layer(g, h)

            # residual connection
            if self.use_linear:
                h_conv = h_conv + linear
            
            concatenated = torch.cat(stack, dim=-1)

            # add aggregation buffer
            h = h_conv + norm * self.layer_buffers[i](concatenated)

            # activation and norm
            if i != len(self.layers) - 1 or self.encoder_mode:
                h = self.activations[i](self.norms_buffer[i](h))
            stack.append(h)
    
        return stack[1:] if return_all else stack[-1]
    
    def transfer_weights(self, trained_model):
        if len(self.layers) != len(trained_model.layers):
            raise ValueError("The number of layers in trained GNN and GNN_B must match for weight transfer.")

        for i, layer in enumerate(self.layers):
            # Transfer weights and bias for GraphConv layers
            layer.weight.data = trained_model.layers[i].weight.data.clone().detach()
            if layer.bias is not None and trained_model.layers[i].bias is not None:
                layer.bias.data = trained_model.layers[i].bias.data.clone().detach()
            
            # Transfer weights for linear layers (if used)
            if self.use_linear and trained_model.use_linear:
                self.linears[i].weight.data = trained_model.linears[i].weight.data.clone()

            # Transfer norm layers if necessary
            if isinstance(self.norms_buffer[i], torch.nn.LayerNorm) and \
               isinstance(trained_model.norms[i], torch.nn.LayerNorm):
                self.norms_buffer[i].weight.data = trained_model.norms[i].weight.data.clone()
                self.norms_buffer[i].bias.data = trained_model.norms[i].bias.data.clone()
                
        print("Transferred weights from trained GNN to GNN_B successfully.")

class SAGE_B(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout=0.5,
                 use_linear=False,
                 norm='identity',
                 prelu=False,
                 encoder_mode=False,
                 mp_norm='both'):
        super(SAGE_B, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms_buffer = torch.nn.ModuleList()
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
        
        # parameters of aggregation buffer
        self.layer_buffers = torch.nn.ModuleList()

        accumulated_dim = in_feats  # Initialize with input features dimension
        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            self.layers.append(dgl.nn.SAGEConv(in_, out_, aggregator_type = "mean"))
            self.norms_buffer.append(norm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))
            
            # Initialize buffer in each layer
            layer_buffer = torch.nn.Linear(accumulated_dim, out_, bias=True)
            torch.nn.init.constant_(layer_buffer.weight, 0.0)
            if layer_buffer.bias is not None:
                torch.nn.init.constant_(layer_buffer.bias, 0.0)
            self.layer_buffers.append(layer_buffer)
            
            # Accumulate dimensions for the next layer
            accumulated_dim += out_

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]

    def forward(self, g, features, return_all=False):
        h = features
        stack = [features]

        # Prepare normalization for layer buffer
        degs = g.in_degrees().float().clamp(min=1) 
        norm = torch.pow(degs, -1).unsqueeze(-1) 

        for i, layer in enumerate(self.layers):
            # dropout
            if i != 0: h = self.dropout(h)

            # apply linear
            if self.use_linear:
                linear = self.linears[i](h)
            # graph conv
            h_conv = layer(g, h)

            # residual connection
            if self.use_linear:
                h_conv = h_conv + linear
            
            concatenated = torch.cat(stack, dim=-1)

            # add aggregation buffer
            h = h_conv + norm * self.layer_buffers[i](concatenated)

            # activation and norm
            if i != len(self.layers) - 1 or self.encoder_mode:
                h = self.activations[i](self.norms_buffer[i](h))
            stack.append(h)
      
        return stack[1:] if return_all else stack[-1]
    
    def transfer_weights(self, trained_model):
        if len(self.layers) != len(trained_model.layers):
            raise ValueError("The number of layers in trained GNN and GNN_B must match for weight transfer.")

        for i, layer in enumerate(self.layers):
            layer.fc_neigh.weight.data = trained_model.layers[i].fc_neigh.weight.data.clone().detach()
            layer.fc_self.weight.data = trained_model.layers[i].fc_self.weight.data.clone().detach()
            if layer.fc_self.bias is not None and trained_model.layers[i].fc_self.bias is not None:
                layer.fc_self.bias.data = trained_model.layers[i].fc_self.bias.data.clone().detach()  

            # Transfer weights for linear layers (if used)
            if self.use_linear and trained_model.use_linear:
                self.linears[i].weight.data = trained_model.linears[i].weight.data.clone()
                
            # Transfer norm layers if necessary
            if isinstance(self.norms_buffer[i], torch.nn.LayerNorm) and \
               isinstance(trained_model.norms[i], torch.nn.LayerNorm):
                self.norms_buffer[i].weight.data = trained_model.norms[i].weight.data.clone()
                self.norms_buffer[i].bias.data = trained_model.norms[i].bias.data.clone()
                
        print("Transferred weights from trained GNN to GNN_B successfully.")

class GAT_B(nn.Module):
    def __init__(self, in_size, hid_size, out_size, use_linear=False, dropout=0.0, heads=[8,1], norm=False):
        super(GAT_B, self).__init__()
        self.layers = nn.ModuleList()
        # two-layer GAT
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
        self.norms_buffer = torch.nn.ModuleList()
        for in_, out_ in zip([in_size, hid_size], [hid_size, out_size]):
            self.norms_buffer.append(norm(out_))
            self.activations.append(torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))

        self.layers.append(
            dgl.nn.GATConv(
                in_size,
                hid_size // heads [0],
                heads[0],
                feat_drop=dropout,
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
                feat_drop=dropout,
                attn_drop=0.3,
                activation=None,
                allow_zero_in_degree=True
            )
        )
 
        # parameters of aggregation buffer
        self.layer_buffers = torch.nn.ModuleList()

        accumulated_dim = in_size  # Initialize with input features dimension
        self.hidden_lst = [in_size, hid_size, out_size]
        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            # Initialize buffer in each layer
            layer_buffer = torch.nn.Linear(accumulated_dim, out_, bias=True)
            torch.nn.init.constant_(layer_buffer.weight, 0.0)
            if layer_buffer.bias is not None:
                torch.nn.init.constant_(layer_buffer.bias, 0.0)
            self.layer_buffers.append(layer_buffer)
            
            # Accumulate dimensions for the next layer
            accumulated_dim += out_

    def forward(self, g, inputs, return_all=False):
        h = inputs
        stack = [inputs]
        # Prepare normalization for layer buffer
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -1).unsqueeze(-1) 

        for i, layer in enumerate(self.layers):
            h_conv = layer(g, h)
            concatenated = torch.cat(stack, dim=-1)

            if i == 1:  # last layer
                h_conv = h_conv.mean(1) 
            else:  # other layer(s)
                h_conv = h_conv.flatten(1) 

            if self.use_linear:
                h_conv = h_conv + self.linears[i](h)
            
            # add aggregation buffer
            h = h_conv + norm * self.layer_buffers[i](concatenated)

            # activation and norm
            if i != len(self.layers) - 1:
                h = self.activations[i](self.norms_buffer[i](h))
            
            stack.append(h)
        return h

    def transfer_weights(self, trained_model):
        if len(self.layers) != len(trained_model.layers):
            raise ValueError("The number of layers in trained GNN and GNN_B must match for weight transfer.")

        for i, layer in enumerate(self.layers):
            # Transfer weights and bias from GraphConv layers
            layer.fc.weight.data = trained_model.layers[i].fc.weight.data.clone().detach()
            layer.attn_l.data = trained_model.layers[i].attn_l.data.clone().detach()
            layer.attn_r.data = trained_model.layers[i].attn_r.data.clone().detach()
            if layer.bias is not None and trained_model.layers[i].bias is not None:
                layer.bias.data = trained_model.layers[i].bias.data.clone().detach()

            # Transfer weights for linear layers (if used)
            if self.use_linear and trained_model.use_linear:
                self.linears[i].weight.data = trained_model.linears[i].weight.data.clone()

            # Transfer norm layers if necessary
            if isinstance(self.norms_buffer[i], torch.nn.LayerNorm) and \
               isinstance(trained_model.norms[i], torch.nn.LayerNorm):
                self.norms_buffer[i].weight.data = trained_model.norms[i].weight.data.clone()
                self.norms_buffer[i].bias.data = trained_model.norms[i].bias.data.clone()
                
        print("Transferred weights from trained GNN to GNN_B successfully.") 

class GIN_B(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout=0.5,
                 use_linear=False,
                 norm='identity',
                 prelu=False,
                 encoder_mode=False,
                 mp_norm='both'):
        super(GIN_B, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms_buffer = torch.nn.ModuleList()
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
        
        # parameters of aggregation buffer
        self.layer_buffers = torch.nn.ModuleList()

        accumulated_dim = in_feats  # Initialize with input features dimension
        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            self.layers.append(dgl.nn.GINConv(torch.nn.Linear(in_, out_)))
            self.norms_buffer.append(norm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))
            
            # Initialize buffer in each layer
            layer_buffer = torch.nn.Linear(accumulated_dim, out_, bias=True)
            torch.nn.init.constant_(layer_buffer.weight, 0.0)
            if layer_buffer.bias is not None:
                torch.nn.init.constant_(layer_buffer.bias, 0.0)
            self.layer_buffers.append(layer_buffer)
            
            # Accumulate dimensions for the next layer
            accumulated_dim += out_

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]

    def forward(self, g, features, return_all=False):
        h = features
        stack = [features]
        # Prepare normalization for layer buffer
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -1).unsqueeze(-1) 

        for i, layer in enumerate(self.layers):
            # dropout
            if i != 0: h = self.dropout(h)

            # apply linear
            if self.use_linear:
                linear = self.linears[i](h)
            # graph conv
            h_conv = layer(g, h)

            # residual connection
            if self.use_linear:
                h_conv = h_conv + linear
            
            concatenated = torch.cat(stack, dim=-1)
            # add aggregation buffer
            h = h_conv + norm * self.layer_buffers[i](concatenated)

            # activation and norm
            if i != len(self.layers) - 1 or self.encoder_mode:
                h = self.activations[i](self.norms_buffer[i](h))
            stack.append(h)
    
        return stack[1:] if return_all else stack[-1]
    
    def transfer_weights(self, trained_model):
        if len(self.layers) != len(trained_model.layers):
            raise ValueError("The number of layers in trained GNN and GNN_B must match for weight transfer.")

        for i, layer in enumerate(self.layers):
            layer.apply_func.weight.data = trained_model.layers[i].apply_func.weight.data.clone().detach()
            if layer.apply_func.bias is not None and trained_model.layers[i].apply_func.bias is not None:
                layer.apply_func.bias.data = trained_model.layers[i].apply_func.bias.data.clone().detach()
        
            # Transfer weights for linear layers (if used)
            if self.use_linear and trained_model.use_linear:
                self.linears[i].weight.data = trained_model.linears[i].weight.data.clone()

            # Transfer norm layers if necessary
                if isinstance(self.norms_buffer[i], torch.nn.LayerNorm) and \
                isinstance(trained_model.norms[i], torch.nn.LayerNorm):
                    self.norms_buffer[i].weight.data = trained_model.norms[i].weight.data.clone()
                    self.norms_buffer[i].bias.data = trained_model.norms[i].bias.data.clone()
                    
            print("Transferred weights from trained GNN to GNN_B successfully.")
  

class SGC_B(torch.nn.Module):
    def __init__(self,
                in_feats,
                hidden_lst,
                dropout=0.0,
                use_linear=False,
                norm='identity',
                prelu=False,
                encoder_mode=False,
                mp_norm='both'):
        super(SGC_B, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.encoder_mode = encoder_mode
        self.hidden_lst = [in_feats, hidden_lst[-1]] 
        self.layers.append(torch.nn.Linear(in_feats, hidden_lst[-1], bias = True))

        # parameters of aggregation buffer
        self.layer_buffers = torch.nn.ModuleList()

        for i in range(2):
            # Initialize buffer in each layer
            layer_buffer = torch.nn.Linear(in_feats + (i+1) * hidden_lst[-1], hidden_lst[-1], bias=True)
            torch.nn.init.constant_(layer_buffer.weight, 0.0)
            if layer_buffer.bias is not None:
                torch.nn.init.constant_(layer_buffer.bias, 0.0)
            self.layer_buffers.append(layer_buffer)

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]

    def forward(self, g, feat, return_all=False):
        degs = g.in_degrees().float().clamp(min=1).to(feat.device).unsqueeze(1)
        norm = torch.pow(degs, -0.5)

        degs = g.in_degrees().float().clamp(min=1) 
        norm_2 = torch.pow(degs, -1).unsqueeze(-1) 

        stack = [feat]
        feat = self.layers[0](feat)
        stack.append(feat)

        for i in range(2):
            # Normalize features
            feat = feat * norm
            g.ndata['h'] = feat
            # Message passing
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
            concatenated = torch.cat(stack, dim=-1)
            feat = g.ndata.pop('h') * norm + norm_2 * (self.layer_buffers[i](concatenated))
 
            stack.append(feat)

        return feat

    def transfer_weights(self, trained_model):
        if len(self.layers) != len(trained_model.layers):
            raise ValueError("The number of layers in GCN and EXT_GCN must match for weight transfer.")

        for i, layer in enumerate(self.layers):
            # Transfer weights and bias from GraphConv layers
            layer.weight.data = trained_model.layers[i].weight.data.clone().detach()
            if layer.bias is not None and trained_model.layers[i].bias is not None:
                layer.bias.data = trained_model.layers[i].bias.data.clone().detach()