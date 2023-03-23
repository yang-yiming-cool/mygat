import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features  # dimension of node vector
        self.out_features = out_features  # dimension of node vector after gat
        self.dropout = dropout  # parameter of dropout
        self.alpha = alpha  # parameter of leakyReLU

        # define W and a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414) 

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_h, adj):
        """
        input_h:  [N, in_features]
        adj: [N, N]
        """
        # self.W [in_features,out_features]
        # input_h × self.W  ->  [N, out_features]

        h = torch.mm(input_h, self.W)  # [N, out_features]

        N = h.size()[0] 
        input_concat = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1). \
            view(N, -1, 2 * self.out_features)
        # [N, N, 2 * out_features]

        e = self.leakyrelu(torch.matmul(input_concat, self.a).squeeze(2))
        # [N, N, 1] => [N, N] correlation coefficient of attention (not normalized)

        zero_vec = -1e12 * torch.ones_like(e)  # set unconnected edges to negative infinity
        
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # if the adjacency matrix element is greater than 0, the two nodes are connected, 
        # and the attention coefficient of this position will be reserved
        # otherwise, mask needs to be set to a very small value, because this value will not be considered in softmax
        attention = F.softmax(attention, dim=1)  # [N, N]
        
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
        output_h = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        
        return output_h
