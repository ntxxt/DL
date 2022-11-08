import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.inits import uniform

in_channels = 2
out_channels = 4
size = 2 * in_channels
#init value as torch parameters
weight = Parameter(torch.Tensor(2*in_channels,out_channels))
#calls Tensor.uniform_ function 
#fills self tensor with numbers sampled from the continuous uniform distribution:
uniform(size, weight)
#  perform || matrix concatenation
alpha = torch.cat([x[row], x[col]], dim=-1)
#scatter mean
'''
Averages all values from the src tensor into out at the indices specified in the index tensor along a given axis dim.I
'''
out = scatter_mean(alpha, col, dim=0, out=out)

#resize 
a = torch.view(1,2)


#GAT code 

import torch
from torch import nn

from labml_helpers.module import Module


class GraphAttentionLayer(Module):
    """
    ## Graph attention layer

    This is a single graph attention layer.
    A GAT is made up of multiple such layers.

    It takes
    $$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$,
    where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input
    and outputs
    $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$,
    where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.

        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # The initial transformation for each head.
        # We do single linear transformation and then split it up for each head.
        # n nodes, each node has m features
        #input h shape (n,m)
        #linear layer output shape (m, output)  
        # m * output = n * heads * hidden 
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden) #[n, head, hidden]

        # #### Calculate attention score
        
        g_repeat = g.repeat(n_nodes, 1, 1) # [n*n, head, hidden]
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0) #[n*n, head, hidden]

        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1) #[2*n*n, head, hidden]
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden) #[n,n,head, 2*hidden]

        # Calculate
        e = self.activation(self.attn(g_concat)) #[2*hidden,1] [n,n,head, 2*hidden] --> [n,n,head,1]
        # Remove the last dimension of size `1`
        e = e.squeeze(-1) #[n,n,head]

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float('-inf')) #only calculate nodes that are connected 
        

        a = self.softmax(e) #normalize attention scores

        # Apply dropout regularization
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g) #a[n,n,head] g[n,head,hidden] --> [n,head,hidden]

        # Concatenate the heads
        if self.is_concat:
            # $$\overrightarrow{h'_i} = \Bigg\Vert_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            # $$\overrightarrow{h'_i} = \frac{1}{K} \sum_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.mean(dim=1)