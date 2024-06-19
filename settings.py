import numpy as np
import scipy.io


class Config:
    pass


class ConfigFlickr(Config):
    # environment and parameters
    data_path = None

    SEMANTIC_EMBED = 512
    batch_size = 128  # bs for test phase
    cpl_batch_size = 64
    icpl_batch_size = 64

    TRAINING_SIZE = 18015
    DATABASE_SIZE = 18015
    QUERY_SIZE = 2000

    FULL = 0.5
    '''LOST_ALL = 1 - FULL
    IMAGE_LOST = 0.5
    TEXT_LOST = 1 - IMAGE_LOST'''

    Epoch = 50
    k_lab_net = 10
    k_img_net = 15
    k_txt_net = 15

    bit = 32
    lr_lab = np.linspace(np.power(10, -2.), np.power(10, -6.), Epoch + 5)
    lr_img = np.linspace(np.power(10, -4.5), np.power(10, -6.), Epoch)
    lr_txt = np.linspace(np.power(10, -3.0), np.power(10, -6.), Epoch)

    def update_from_args(self, args):
        self.data_path = args.data_path
        self.SEMANTIC_EMBED = args.encode_feature_dim
        self.cpl_batch_size = args.batch_size_train
        self.batch_size = args.batch_size_test
        self.TRAINING_SIZE = args.n_dset
        self.DATABASE_SIZE = args.n_dset
        self.QUERY_SIZE = args.n_query
        self.FULL = args.full_ratio
        self.LOST_ALL = 1 - self.FULL
        self.IMAGE_LOST = 0.5
        self.TEXT_LOST = 1 - self.IMAGE_LOST
        self.Epoch = args.epoch
        self.bit = args.bit

cfg = ConfigFlickr()
