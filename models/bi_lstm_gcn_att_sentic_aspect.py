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
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class BI_LSTM_GCN_ATT_SENTIC_ASPECT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(BI_LSTM_GCN_ATT_SENTIC_ASPECT, self).__init__()        
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        
        self.bilstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        self.attention1 = Attention(2 * opt.hidden_dim, score_function='bi_linear')
        self.attention2 = Attention(2 * opt.hidden_dim, score_function='bi_linear')
        self.attention3 = Attention(2 * opt.hidden_dim, score_function='bi_linear')
        
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):       
        text_indices, aspect_indices, sdat_graph = inputs
        
        embedded_text = self.embed(text_indices)
        embedded_aspect = self.embed(aspect_indices)
        
        combined = embedded_text + embedded_aspect
        
        x_len = torch.sum(text_indices != 0, dim=-1)
        
        lstm_out, (h_n, _) = self.bilstm(combined, x_len)

        seq_len = lstm_out.size(1)
        sdat_graph = sdat_graph[:, :seq_len, :seq_len]
        
        gcn_output = F.relu(self.gc1(lstm_out, sdat_graph))
        gcn_output = F.relu(self.gc2(gcn_output, sdat_graph))

        attn_output, attn_weights = self.attention1(gcn_output, gcn_output)
        attn_output, attn_weights = self.attention2(attn_output, attn_output)
        attn_output, attn_weights = self.attention3(attn_output, attn_output)
        
        pooled_features = torch.mean(attn_output, dim=1)
        
        dropped_features = self.dropout(pooled_features)
        
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
    