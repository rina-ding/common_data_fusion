
# Copied and modified from https://github.com/irsyadadam/MMCANetwork_torch/blob/main/MMCA/cross_attn.py

from typing import Optional, Tuple, Union

import torch

class cross_attn_block(torch.nn.Module):
    r"""
    Single Block for Cross Attention

    Args: 
        m1: first modality
        m2: second modality

    Shapes: 
        m1: (seq_length, N_samples, N_features)
        m2: (seq_length, N_samples, N_features)

    Returns: 
        embedding of m1 depending on attending on certain elements of m2, multihead_attn(k_m1, v_m1, q_m2)
    """

    def __init__(self, 
                 dim: int, 
                 heads: int, 
                 dropout: float, 
                 add_positional: Optional[bool] = False):

        super(cross_attn_block, self).__init__()

        #learnable
        self._to_key = torch.nn.Linear(dim, dim)
        self._to_query = torch.nn.Linear(dim, dim)
        self._to_value = torch.nn.Linear(dim, dim)

        self.attn = torch.nn.MultiheadAttention(embed_dim = dim, num_heads = heads, dropout = dropout, batch_first=True)

    def forward(self, 
                m1_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                m2_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        m2_q = self._to_query(m2_x)
        m1_k = self._to_key(m1_x)
        m1_v = self._to_value(m1_x)

        #crossing
        # cross_x, attn_weights = self.attn(m1_k, m1_v, m2_q)
        cross_x, attn_weights = self.attn(m2_q, m1_k, m1_v)

        return cross_x

class cross_attn_channel(torch.nn.Module):
    r"""
    Model for Cross Attention, architecture implementation taken from encoder layer of "Attention is all you need"
    Includes multi-head attn with crossing --> add + norm --> positionwise ffn --> add + norm --> output (based on paper)
    """

    def __init__(self, 
                 encoder_name: str,
                 dim_m1: int, 
                 dim_m2: int, 
                 pffn_dim: int,
                 heads: Optional[int], 
                 seq_len: int, 
                 dropout: float = 0.0):
        r"""
        ARGS: 
            dim_m1: dim of representations of m1
            dim_m2: dim of representations of m2
            pffn_dim: dim of hidden layer of positional-wise ffn
            heads: number of heads for multi-head attn
            seq_len: length of seq
            dropout: dropout rate

        """
        super(cross_attn_channel, self).__init__()
        self.encoder_name = encoder_name
        if 'one_way_attention' in self.encoder_name:
            self.m1_cross_m2 = cross_attn_block(dim = dim_m1, heads = heads, dropout = dropout)

        else:
            self.m1_cross_m2 = cross_attn_block(dim = dim_m1, heads = heads, dropout = dropout)
            self.m2_cross_m1 = cross_attn_block(dim = dim_m2, heads = heads, dropout = dropout)

        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, 
                m1: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                m2: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        r"""

        ARGS: 
            m1: input tensor of (seq_length, N_samples, N_features)
            m2: input tensor of (seq_length, N_samples, N_features)
            mask: NOT IMPLEMENTED

        RETURNS:
            tranformed m1, m2, with output dim same as input dim, but with attention
            
        """
        if 'one_way_attention' in self.encoder_name:
            m1_x = self.m1_cross_m2(m1, m2)
            m2_x = m2

        else:
            m1_x = self.m1_cross_m2(m1, m2)
            m2_x = self.m2_cross_m1(m2, m1)

        m1_x = self.dropout(m1_x)
        m2_x = self.dropout(m2_x)

        return m1_x, m2_x
