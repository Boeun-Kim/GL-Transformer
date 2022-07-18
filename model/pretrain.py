import torch
import torch.nn as nn
from model.transformer import ST_Transformer
from einops import rearrange
import os

class DirPred(nn.Module):
    def __init__(self, num_joint, num_frame, dim_emb):
        super(DirPred, self).__init__()
        self.num_joint = num_joint
        self.num_frame = num_frame

        self.linear = nn.Sequential(
                                    nn.Linear(num_joint*dim_emb, num_joint*27)
                                    )
    def forward(self, x):
        x = self.linear(x)
        x = rearrange(x, 'b f (j k) -> b k f j', j=self.num_joint)

        return x

class MagPred(nn.Module):
    def __init__(self, num_joint, num_frame, dim_emb):
        super(MagPred, self).__init__()
        self.num_joint = num_joint
        self.num_frame = num_frame

        self.linear = nn.Sequential(
                                    nn.Linear(num_joint*dim_emb, num_joint*16)
                                    )
    def forward(self, x):
        x = self.linear(x)
        x = rearrange(x, 'b f (j k) -> b k f j', j=self.num_joint)
        return x


class Pretrain(nn.Module):

    def __init__(self, num_frame, num_joint, input_channel, dim_emb,
                depth, num_heads, qkv_bias, ff_expand, do_rate, attn_do_rate,
                drop_path_rate, add_positional_emb, intervals):
        super(Pretrain, self).__init__()

        self.num_joint=num_joint

        self.transformer = ST_Transformer(num_frame, num_joint, input_channel, dim_emb,
                                        depth, num_heads, qkv_bias, ff_expand, do_rate, attn_do_rate,
                                        drop_path_rate, add_positional_emb, positional_emb_type='learnable')

        self.dir_pred_list = nn.ModuleList(
            [DirPred(num_joint, num_frame, dim_emb) for i in range(len(intervals))]
        )

        self.mag_pred_list = nn.ModuleList(
            [MagPred(num_joint, num_frame, dim_emb) for i in range(len(intervals))]
        )

    def forward(self, position):

        x = self.transformer(position)
        
        ## predict the direction of displacements
        out_dir_list = []
        for dir_pred in self.dir_pred_list:
            out_dir_list.append(dir_pred(x))

        out_dir = torch.stack(out_dir_list)
        out_dir = rearrange(out_dir, 'r b c f j  -> b c f (r j)', )

        ## predict the magnitude of displacements
        out_mag_list = []
        for mag_pred in self.mag_pred_list:
            out_mag_list.append(mag_pred(x))

        out_mag = torch.stack(out_mag_list)
        out_mag = rearrange(out_mag, 'r b c f j  -> b c f (r j)', )

        return out_dir, out_mag


    def save(self, epoch, file_path = ""):
        os.makedirs(file_path, exist_ok=True)
        output_path = file_path + "/epoch%d" % epoch
        torch.save(self.transformer.state_dict(), output_path)