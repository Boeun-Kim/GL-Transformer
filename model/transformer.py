import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=150):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Attention(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0., proj_do_rate=0.):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.qkv = nn.Linear(dim_emb, dim_emb * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, mask=None):

        B, N, C = x.shape  

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, do_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(do_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_joint=50, num_frame=150, dim_emb=48, 
                num_heads=8, ff_expand=1.0, qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0., positional_emb_type='learnalbe'):

        super(TransformerEncoder, self).__init__()

        self.positional_emb_type = positional_emb_type

        # for learnable positional embedding
        self.positional_emb = nn.Parameter(torch.zeros(1, num_frame, num_joint, dim_emb))

        # for fixed positional embedding (ablation)
        self.tm_pos_encoder = PositionalEmbedding(num_joint*dim_emb, num_frame)
        self.sp_pos_encoder = PositionalEmbedding(dim_emb, num_joint)

        self.norm1_sp = nn.LayerNorm(dim_emb)
        self.norm1_tm = nn.LayerNorm(dim_emb*num_joint)

        self.attention_sp = Attention(dim_emb, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
        self.attention_tm = Attention(dim_emb*num_joint, num_heads, qkv_bias, attn_do_rate, proj_do_rate)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim_emb*num_joint)
        self.feedforward = FeedForward(in_features=dim_emb*num_joint, hidden_features=int(dim_emb*num_joint*ff_expand), 
                                        out_features=dim_emb*num_joint, do_rate=proj_do_rate)
                            

    def forward(self, x, mask=None, positional_emb=False):

        b, f, j, c = x.shape

        ## spatial-MHA
        x_sp = rearrange(x, 'b f j c  -> (b f) j c', )
        if positional_emb==True:
            if self.positional_emb_type=='fix':
                x_sp = x_sp + self.sp_pos_encoder(x_sp)
            else:
                pos_emb = self.positional_emb.repeat(b, 1,1,1)
                pos_emb = rearrange(pos_emb, 'b f j c -> (b f) j c', b=b, f=f)
                x_sp = x_sp + pos_emb

        x_sp = x_sp + self.drop_path(self.attention_sp(self.norm1_sp(x_sp), mask=None))
  
        ## temporal-MHA
        x_tm = rearrange(x_sp, '(b f) j c -> b f (j c)', b=b, f=f)
        if positional_emb==True:
            if self.positional_emb_type=='fix':
                x_tm = x_tm + self.tm_pos_encoder(x_tm)
            else:
                pos_emb = rearrange(pos_emb, '(b f) j c -> b f (j c)', b=b, f=f)
                x_tm = x_tm + pos_emb

        x_tm = x_tm + self.drop_path(self.attention_tm(self.norm1_tm(x_tm), mask=mask))

        x_out = x_tm
        x_out = x_out + self.drop_path(self.feedforward(self.norm2(x_out)))
        x_out = rearrange(x_out, 'b f (j c)  -> b f j c', j=j)

        return x_out

class ST_Transformer(nn.Module):

    def __init__(self, num_frame, num_joint, input_channel, dim_joint_emb,
                depth, num_heads, qkv_bias, ff_expand, do_rate, attn_do_rate,
                drop_path_rate, add_positional_emb, positional_emb_type):

        super(ST_Transformer, self).__init__()

        self.num_joint = num_joint
        self.num_frame = num_frame
        self.add_positional_emb = add_positional_emb
        
        self.dropout = nn.Dropout(p=do_rate)
        self.norm = nn.LayerNorm(dim_joint_emb*num_joint)

        self.emb = nn.Linear(input_channel, dim_joint_emb)
        self.emb_global = nn.Linear(input_channel, dim_joint_emb)
        self.pred_token_emb = nn.Parameter(torch.zeros(1, 1, num_joint, dim_joint_emb))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(num_joint, num_frame, dim_joint_emb, 
            num_heads, ff_expand, qkv_bias, attn_do_rate, do_rate, dpr[i], positional_emb_type) 
            for i in range(depth)]
        )
        
        self.mlp = nn.Sequential(
                                nn.Linear(dim_joint_emb*num_joint, dim_joint_emb*num_joint),
                                nn.GELU(),
                                nn.LayerNorm(dim_joint_emb*num_joint),
                                nn.Linear(dim_joint_emb*num_joint, dim_joint_emb*num_joint),
                                nn.GELU(),
                                nn.LayerNorm(dim_joint_emb*num_joint),
                                )

    def encoder(self, x, mask=None):

        b, c, f, j = x.shape

        ## generate input embeddings
        x = rearrange(x, 'b c f (j p) -> b c f j p', p=2)
        x_joints = torch.cat((x[:,:,:,:20,:], x[:,:,:,21:,:]), axis=3)
        x_global = x[:,:,:,20,:].unsqueeze(3)

        x_joints = rearrange(x_joints, 'b c f j p -> b f j p c')
        x_global = rearrange(x_global, 'b c f j p -> b f j p c')
        x_joints = self.emb(x_joints)  # joint embedding layer
        x_global = self.emb_global(x_global)  # global translation embedding layer

        x = torch.cat((x_joints[:,:,:20,:,:], x_global, x_joints[:,:,20:,:,:]), axis=2)
        x = rearrange(x, 'b f j p c-> b f (j p) c',)
        
        x = self.dropout(x)

        ## GL-Transformer blocks
        for i, block in enumerate(self.encoder_blocks):
            if self.add_positional_emb:
                positional_emb=True
            else:
                positional_emb = False
            x = block(x, mask, positional_emb)

        x = rearrange(x, 'b f j k -> b f (j k)',j=j)
        x = self.norm(x)

        return x


    def forward(self, x):

        ## make attention mask for [PAD] tokens
        x_m = x[:,0,:,0]
        mask = (x_m != 99.9).unsqueeze(1).repeat(1, x_m.size(1), 1).unsqueeze(1)

        x = self.encoder(x, mask)

        ## MLP
        x = self.mlp(x) 

        return x
        
