# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention

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


class BI_LSTM_GCN_ATT_SENTIC_DEP_ASPECT_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(BI_LSTM_GCN_ATT_SENTIC_DEP_ASPECT_BERT, self).__init__()
        self.bert = bert
        
        self.bilstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        self.attention1 = Attention(2 * opt.hidden_dim, score_function='bi_linear')
        self.attention2 = Attention(2 * opt.hidden_dim, score_function='bi_linear')
        self.attention3 = Attention(2 * opt.hidden_dim, score_function='bi_linear')
        self.attention4 = Attention(2 * opt.hidden_dim, score_function='bi_linear')

        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        text_bert_indices, text_indices, bert_segments_ids, aspect_indices, dependency_graph, sdat_graph = inputs
        
        text_bert, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        aspect_bert, _ = self.bert(aspect_indices, token_type_ids=torch.zeros_like(aspect_indices), output_all_encoded_layers=False)
        
        combined = text_bert + aspect_bert
        
        x_len = torch.sum(text_indices != 0, dim=-1)
        
        lstm_out, (h_n, _) = self.bilstm(combined, x_len)
        
        seq_len = lstm_out.size(1)
        dependency_graph = dependency_graph[:, :seq_len, :seq_len]
        sdat_graph = sdat_graph[:, :seq_len, :seq_len]
        
        combined_graph = dependency_graph * sdat_graph
        
        gcn_output = F.relu(self.gc1(lstm_out, combined_graph))
        gcn_output = F.relu(self.gc2(gcn_output, combined_graph))
        
        attn_output1, attn_weights1 = self.attention1(gcn_output, gcn_output)
        attn_output2, attn_weights2 = self.attention2(gcn_output, gcn_output)
        attn_output3, attn_weights3 = self.attention3(gcn_output, gcn_output)
        attn_output4, attn_weights4 = self.attention4(gcn_output, gcn_output)

        attn_output = (
            attn_output1 +
            attn_output2 +
            attn_output3 +
            attn_output4
        ) / 4.0
        
        pooled_features = torch.mean(attn_output, dim=1)
        
        dropped_features = self.dropout(pooled_features)
        
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
