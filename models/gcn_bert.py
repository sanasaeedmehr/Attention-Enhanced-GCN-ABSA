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
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(GCN_BERT, self).__init__()
        self.bert = bert
        
        self.gc1 = GraphConvolution(opt.bert_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, dependency_graph = inputs
        
        encoder_layer, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        
        x = F.relu(self.gc1(encoder_layer, dependency_graph))
        x = F.relu(self.gc2(x, dependency_graph))
        
        pooled_features = torch.mean(x, dim=1)
        
        dropped_features = self.dropout(pooled_features)
        
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
        

