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

class GCN_ASPECT_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(GCN_ASPECT_BERT, self).__init__()
        self.bert = bert
        
        self.gc1 = GraphConvolution(opt.bert_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        self.dropout_layer = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, aspect_indices, dependency_graph = inputs
        
        text_repr, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        aspect_repr, _ = self.bert(aspect_indices, token_type_ids=torch.zeros_like(aspect_indices), output_all_encoded_layers=False)
        
        # aspect_representation = torch.mean(aspect_layer, dim=1)  # Shape: [batch_size, bert_dim]

        # aspect_transformed = F.relu(self.aspect_fc(aspect_representation))  # Shape: [batch_size, bert_dim]

        # aspect_expanded = aspect_transformed.unsqueeze(1).expand_as(encoder_layer)  # Shape: [batch_size, seq_len, bert_dim]
        # combine = encoder_layer + aspect_expanded  # Shape: [batch_size, seq_len, bert_dim]
        
        combined_repr = text_repr + aspect_repr
        
        gcn_output = F.relu(self.gc1(combined_repr, dependency_graph))
        gcn_output = F.relu(self.gc2(gcn_output, dependency_graph))
        
        pooled_features = torch.mean(gcn_output, dim=1)
        
        dropped_features = self.dropout(pooled_features)
        
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
