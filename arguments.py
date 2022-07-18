import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training Transformer')

    # train hyperparameters
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.00005) 
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--num_workers', type=int, default=8)

    # augmentation
    parser.add_argument('-s', '--shear_amplitude', type=float, default=0.3)
    parser.add_argument('-i', '--interpolate_ratio', type=float, default=0.1)

    # multi-interval displacement prediction & loss parameters
    parser.add_argument('--intervals', type=int, nargs='+', default=[1,5,10])
    parser.add_argument('--lambda_mag', type=float, default=1.0)
    parser.add_argument('--lambda_global', type=float, default=0.05)

    # model parameters
    parser.add_argument('--num_frame', type=int, default=300)
    parser.add_argument('--num_joint', type=int, default=50)
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--dim_emb', type=int, default=48)

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--qkv_bias', type=bool, default=True)
    parser.add_argument('--ff_expand', type=int, default=2.0)

    parser.add_argument('--do_rate', type=float, default=0.1)
    parser.add_argument('--attn_do_rate', type=float, default=0.1)

    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--add_positional_emb', type=int, default=1)

    # path
    parser.add_argument('--train_data_path', type=str, default="data/NTU60-preprocessed/xsub/train_position.npy")
    parser.add_argument('--eval_data_path', type=str, default="data/NTU60-preprocessed/xsub/val_position.npy")
    parser.add_argument('--train_label_path', type=str, default="data/NTU-RGB-D60/xsub/train_label.pkl")
    parser.add_argument('--eval_label_path', type=str, default="data/NTU-RGB-D60/xsub/val_label.pkl")
    parser.add_argument('--save_path', type=str, default="experiment/pretrained")

    args = parser.parse_args()

    return args


def parse_args_actionrecog():
    parser = argparse.ArgumentParser(description='Training Transformer')

    # train hyperparameters
    parser.add_argument('--gpus', type=int, default=4) 
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--num_workers', type=int, default=8)

    # augmentation
    parser.add_argument('-s', '--shear_amplitude', type=float, default=-1)
    parser.add_argument('-i', '--interpolate_ratio', type=float, default=0.1)

    # model parameters
    parser.add_argument('--pretrained_model', type=str, default="experiment/pretrained/epoch1")
    parser.add_argument('--pretrained_model_w_classifier', type=str, default="")

    parser.add_argument('--num_frame', type=int, default=300)
    parser.add_argument('--num_joint', type=int, default=50)
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--dim_emb', type=int, default=48)

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--qkv_bias', type=bool, default=True)
    parser.add_argument('--ff_expand', type=float, default=2.0)

    parser.add_argument('--do_rate', type=float, default=0.1)
    parser.add_argument('--attn_do_rate', type=float, default=0.1)

    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--add_positional_emb', type=int, default=1)
    parser.add_argument('--positional_emb_type', type=str, default='learnable') #learnable, fix

    parser.add_argument('--num_action_class', type=int, default=60)

    # path
    parser.add_argument('--train_data_path', type=str, default="data/NTU60-preprocessed/xsub/train_position.npy")
    parser.add_argument('--eval_data_path', type=str, default="data/NTU60-preprocessed/xsub/val_position.npy")
    parser.add_argument('--train_label_path', type=str, default="data/NTU-RGB-D60/xsub/train_label.pkl")
    parser.add_argument('--eval_label_path', type=str, default="data/NTU-RGB-D60/xsub/val_label.pkl")
    parser.add_argument('--save_path', type=str, default="experiment/action_recog")

    args = parser.parse_args()

    return args
