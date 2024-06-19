import os
import h5py
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class AnyModelDataset(Dataset):
    def __init__(self, modal_l, labels=None, ind_shift=0, shuffle=False):
        self.modals = modal_l
        self.labels = labels
        self.num = len(self.labels)
        self.n_modal = len(self.modals)
        self.ind_shift = ind_shift
        self.shuffle = shuffle

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        if self.shuffle:
            index = np.random.randint(self.num)
        ret = [index + self.ind_shift]
        for modal in self.modals:
            ret.append(modal[index])
        ret.append(self.labels[index])
        return ret


def get_dataloader(modal_l, labels, bs, ind_shift=0, shuffle=True, drop_last=False):
    dset = AnyModelDataset(modal_l, labels, ind_shift=ind_shift, shuffle=shuffle)
    loader = DataLoader(dset, batch_size=bs, shuffle=False, pin_memory=True, drop_last=drop_last)
    return loader


def get_all_dataloaders(cfg):
    X, Y, L = load_data(cfg)
    loaders = {
        'qloader': get_dataloader([X['query'], Y['query']], L['query'], cfg.batch_size, shuffle=False),
        'vrloader': get_dataloader([X['retrieval_v']], L['retrieval_v'], cfg.batch_size, shuffle=False),
        'trloader': get_dataloader([Y['retrieval_t']], L['retrieval_t'], cfg.batch_size, shuffle=False),
        'floader': get_dataloader([X['full'], Y['full']], L['full'], cfg.cpl_batch_size),
        'vloader': get_dataloader([X['icpl_v']], L['icpl_v'], cfg.cpl_batch_size, ind_shift=cfg.num_f),
        'tloader': get_dataloader([Y['icpl_t']], L['icpl_t'], cfg.cpl_batch_size, ind_shift=cfg.num_f + cfg.num_v),
        'lloader': get_dataloader([], L['train'], cfg.cpl_batch_size),
    }
    orig_data = {
        'X': X,
        'Y': Y,
        'L': L
    }
    return loaders, orig_data


def load_data(cfg):
    data_path = cfg.data_path
    file = h5py.File(os.path.join(data_path, 'IAll/mirflickr25k-iall.mat'))
    images = (file['IAll'][:].transpose(0, 1, 3, 2) / 255.0).astype(np.float32)
    tags = loadmat(os.path.join(data_path, 'YAll/mirflickr25k-yall.mat'))['YAll'].astype(np.float32)
    labels = loadmat(os.path.join(data_path, 'LAll/mirflickr25k-lall.mat'))['LAll'].astype(np.float32)
    file.close()

    QUERY_SIZE = cfg.QUERY_SIZE
    TRAINING_SIZE = cfg.TRAINING_SIZE
    FULL = int(cfg.FULL * TRAINING_SIZE)
    LOST_ALL = TRAINING_SIZE - FULL
    IMAGE_LOST = int(cfg.IMAGE_LOST * LOST_ALL)
    TEXT_LOST = LOST_ALL - IMAGE_LOST
    cfg.num_f = FULL
    cfg.num_lost = LOST_ALL
    cfg.num_v = IMAGE_LOST
    cfg.num_t = TEXT_LOST
    cfg.numClass = labels.shape[1]
    cfg.dimTxt = tags.shape[1]

    X = {}
    np.random.seed(0)
    index_all = np.random.permutation(QUERY_SIZE + TRAINING_SIZE)

    images = images[index_all, :, :, :]

    X['full'] = images[:FULL, :, :, :]
    X['icpl_v'] = images[FULL:FULL + IMAGE_LOST, :, :, :]
    X['query'] = images[TRAINING_SIZE:TRAINING_SIZE + QUERY_SIZE, :, :, :]
    X['retrieval_v'] = np.concatenate([X['full'], X['icpl_v']], axis=0)

    Y = {}
    tags = tags[index_all, :]
    Y['full'] = tags[:FULL, :]
    Y['icpl_t'] = tags[FULL + IMAGE_LOST:FULL + IMAGE_LOST + TEXT_LOST, :]
    Y['query'] = tags[TRAINING_SIZE:TRAINING_SIZE + QUERY_SIZE, :]
    Y['retrieval_t'] = np.concatenate([Y['full'], Y['icpl_t']], axis=0)

    L = {}
    labels = labels[index_all, :]

    L['train'] = labels[:TRAINING_SIZE, :]
    L['full'] = labels[:FULL]
    L['icpl_v'] = L['train'][FULL:FULL + IMAGE_LOST]
    L['icpl_t'] = L['train'][FULL + IMAGE_LOST:FULL + IMAGE_LOST + TEXT_LOST]
    L['query'] = labels[TRAINING_SIZE:TRAINING_SIZE + QUERY_SIZE, :]
    L['retrieval_v'] = np.concatenate([L['full'], L['icpl_v']], axis=0)
    L['retrieval_t'] = np.concatenate([L['full'], L['icpl_t']], axis=0)
    #L['retrieval'] = L['full']
    return X, Y, L

'''
def split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE):
    X = {}
    index_all = np.random.permutation(QUERY_SIZE+DATABASE_SIZE)
    ind_Q = index_all[0:QUERY_SIZE]
    ind_T = index_all[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE]
    ind_R = index_all[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE]

    X['query'] = images[ind_Q, :, :, :]
    X['train'] = images[ind_T, :, :, :]
    X['retrieval'] = images[ind_R, :, :, :]

    Y = {}
    Y['query'] = tags[ind_Q, :]
    Y['train'] = tags[ind_T, :]
    Y['retrieval'] = tags[ind_R, :]

    L = {}
    L['query'] = labels[ind_Q, :]
    L['train'] = labels[ind_T, :]
    L['retrieval'] = labels[ind_R, :]
    return X, Y, L
'''

if __name__ == "__main__":
    import pickle
    from settings import cfg
    X, Y, L= load_data(cfg)
    with open('data_processed_ours.pkl', 'wb') as f:
        pickle.dump([X, Y, L], f)
