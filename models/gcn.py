# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)

        # Symmetric normalization with absolute degree and epsilon
        # A_norm = D_eps^(-1/2) A D_eps^(-1/2)
        eps = 1e-6

        # Absolute degree for signed weighted graphs
        degree = torch.sum(torch.abs(adj), dim=2)
        degree_eps = degree + eps

        # D^(-1/2)
        d_inv_sqrt = torch.pow(degree_eps, -0.5)

        # Symmetric normalization
        adj_norm = adj * d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)

        output = torch.matmul(adj_norm, hidden)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(GCN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        
        self.gc1 = GraphConvolution(opt.embed_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        text_indices, dependency_graph = inputs
        
        embedded_text = self.embed(text_indices)
        
        features_after_gcn1 = F.relu(self.gc1(embedded_text, dependency_graph))
        features_after_gcn2 = F.relu(self.gc2(features_after_gcn1, dependency_graph))
        
        pooled_features = torch.mean(features_after_gcn2, dim=1)
        
        dropped_features = self.dropout(pooled_features)
        
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
        