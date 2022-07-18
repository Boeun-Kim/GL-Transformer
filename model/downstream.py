import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import ST_Transformer
from einops import rearrange, repeat
import os

class ActionRecognition(nn.Module):

    def __init__(self, num_frame, num_joint, input_channel, dim_emb,
                depth, num_heads, qkv_bias, ff_expand, do_rate, attn_do_rate,
                drop_path_rate, add_positional_emb, 
                num_action_class, positional_emb_type):
        super(ActionRecognition, self).__init__()

        self.num_joint = num_joint
        self.transformer = ST_Transformer(num_frame, num_joint, input_channel, dim_emb,
                                            depth, num_heads, qkv_bias, ff_expand, do_rate, attn_do_rate,
                                            drop_path_rate, add_positional_emb, positional_emb_type)

        self.tm_pooling = nn.AvgPool1d(num_frame)
        self.linear = nn.Linear(num_joint*dim_emb, num_action_class)


    def forward(self, position):

        x = position
        x_m = position[:,0,:,0] != 99.9
        x = self.transformer(x)  # b, f, jxc
        
        x_out = []
        for b_idx, x_b in enumerate(x):

            x_tmep = x_b[x_m[b_idx]]
            if x_tmep.shape[0] <= 1:
                x_tmep = x_b[0]
            else:
                tm_pooling = nn.AvgPool1d(x_tmep.shape[0])
                x_tmep = rearrange(x_tmep, 'f jc -> jc f',)
                x_tmep = tm_pooling(x_tmep.unsqueeze(0)).squeeze(0).squeeze(1)

            x_out.append(x_tmep)
            
        x_out = torch.stack(x_out)
        x_out = self.linear(x_out)

        return x_out


    def load_transformer(self, file_path = "experiment/pretrained/epoch120"):
        
        load_path = file_path

        self.transformer.load_state_dict(torch.load(load_path))

        # freeze pretrained model
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

