'''
test_actionrecog.py is for testing action recognition (transformer+linear classifier) in the validation set.
'''

import os
import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pickle

from feeder.ntu_feeder import Feeder_actionrecog
from model.downstream import ActionRecognition

from arguments import parse_args_actionrecog

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()


def load_data(is_train, data_path, label_path, batch_size, num_workers, shear_amplitude=-1, interpolate_ratio=-1):
    train_feeder_args = {'data_path': data_path,
                        'label_path': label_path,
                        'shear_amplitude': shear_amplitude,
                        'interpolate_ratio': interpolate_ratio,
                        'mmap': True
                        }
    test_feeder_args = {'data_path': data_path,
                        'label_path': label_path,
                        'shear_amplitude': -1,
                        'interpolate_ratio': -1,
                        'mmap': True
                        }

    if is_train:
        data_loader = torch.utils.data.DataLoader(
            dataset=Feeder_actionrecog(**train_feeder_args),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=init_seed
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=Feeder_actionrecog(**test_feeder_args),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            worker_init_fn=init_seed
        )
        
    return data_loader


def show_best(k, result, label):
    rank = result.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
    accuracy = round(accuracy, 5)
    print("accuracy: ", accuracy)
    return accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    args = parse_args_actionrecog()
    eval_loader = load_data(is_train=False, data_path=args.eval_data_path, label_path=args.eval_label_path, batch_size=args.batch_size, num_workers=args.num_workers,
                        shear_amplitude=args.shear_amplitude, interpolate_ratio=args.interpolate_ratio)
    downstream_model = ActionRecognition
    model = downstream_model(num_frame=args.num_frame, num_joint=args.num_joint, input_channel=args.input_channel, dim_emb=args.dim_emb, 
                            depth=args.depth, num_heads=args.num_heads, qkv_bias=args.qkv_bias, ff_expand=args.ff_expand, 
                            do_rate=args.do_rate ,attn_do_rate=args.attn_do_rate,
                            drop_path_rate=args.drop_path_rate, add_positional_emb=args.add_positional_emb,
                            num_action_class=args.num_action_class, positional_emb_type='learnable')
    model = model.to(device)
    model.load_state_dict(torch.load(args.pretrained_model_w_classifier))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print("gpus: ", np.arange(args.gpus))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=np.arange(args.gpus).tolist())

    criterion = nn.CrossEntropyLoss()

    loss_value = []
    result_frag = []
    label_frag = []

    for position, label in eval_loader:
        print(".")
        position = position.float().to(device)
        label = label.to(device)

        out = model(position)
        result_frag.append(out.data.cpu().numpy())
        label_frag.append(label.data.cpu().numpy())

    result = np.concatenate(result_frag)
    label = np.concatenate(label_frag)
    acc = show_best(1, result, label)


    print("action recognition accuracy : ", acc)
