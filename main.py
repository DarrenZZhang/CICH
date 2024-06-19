import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from settings import cfg
import torch
import numpy as np
from multiprocessing import freeze_support
from trainer import TrainerV0
import argparse


def main():
    trainer = TrainerV0(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=r'D:/MICH/v.resnet/Flicker', type=str,
                        help="image directory for the dataset")
    parser.add_argument("--encode_feature_dim", type=int, default=512, help="common feature size")
    parser.add_argument("--batch_size_train", type=int, default=64, help="batch size used for training")
    parser.add_argument("--batch_size_test", type=int, default=128, help="batch size used to acquire query codes")
    parser.add_argument("--n_dset", type=int, default=18015, help="dataset/training size")
    parser.add_argument('--n_query', type=int, default=2000, help='query/test size')
    parser.add_argument('--full_ratio',  type=float, default=0.5, help='proportion of the paired part')
    parser.add_argument("--epoch", type=int, default=50, help='number of training iterations')
    parser.add_argument("--bit", type=int, default=50, help='number of hash bits / hash code length')
    args = parser.parse_args()
    cfg.update_from_args(args)

    torch.manual_seed(1214090112858600)
    torch.cuda.manual_seed(295516382103593)
    np.random.seed(seed=1234568983)
    freeze_support()
    main()
