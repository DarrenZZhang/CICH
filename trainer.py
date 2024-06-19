from load_data import get_all_dataloaders
from models import ImageNetMI, TextNetMI, LabelNet
from ops import calc_neighbor, adjust_learning_rate
from utils.calc_hammingranking import calc_map
import os
import time
from scipy.io import savemat, loadmat
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np


class TrainerV0:
    def __init__(self, cfg):
        self.cfg = cfg

        # hyper parameters
        self.ic_sel_num = 64
        self.hyper_mi = 50
        self.hyper_sigma = 0.01
        self.alpha_v = 0.01
        self.alpha_t = 1 - self.alpha_v
        self.eta = 100
        self.gamma = 0.01

        # load data
        self.loaders, self.orig_data = get_all_dataloaders(cfg)

        # training configurations
        self.Epoch = cfg.Epoch
        self.lr_lab = cfg.lr_lab
        self.lr_img = cfg.lr_img
        self.lr_txt = cfg.lr_txt
        self.k_lab_net = cfg.k_lab_net
        self.k_img_net = cfg.k_img_net
        self.k_txt_net = cfg.k_txt_net

        # models
        self.inet = ImageNetMI(cfg).cuda()
        self.tnet = TextNetMI(cfg).cuda()
        self.lnet = LabelNet(cfg).cuda()
        self.lnet.train()
        self.inet.train()
        self.tnet.train()

        self.lnet_opt = torch.optim.Adam(self.lnet.parameters(), lr=self.lr_lab[0])
        self.inet_opt = torch.optim.Adam(self.inet.parameters(), lr=self.lr_img[0])
        self.tnet_opt = torch.optim.Adam(self.tnet.parameters(), lr=self.lr_txt[0])

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # incompleteness
        self.num_full = cfg.num_f
        self.num_lost = cfg.num_lost
        self.num_vi = cfg.num_v
        self.num_ti = cfg.num_t
        self.num_train = self.num_full + self.num_vi + self.num_ti
        self.num_full_v = self.num_full + self.num_vi

        # other configurations
        self.bit = cfg.bit
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.batch_size = cfg.batch_size
        self.cpl_batch_size = cfg.cpl_batch_size
        self.icpl_batch_size = cfg.icpl_batch_size


    def train(self):
        var = {}
        var['v'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['vc'] = var['v'][:self.num_full]
        var['vi'] = var['v'][self.num_full: self.num_full_v]
        var['vg'] = var['v'][self.num_full_v:]

        var['t'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['tc'] = var['t'][:self.num_full]
        var['tg'] = var['t'][self.num_full: self.num_full_v]
        var['ti'] = var['t'][self.num_full_v:]

        var['l'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['lc'] = var['l'][:self.num_full]
        var['lv'] = var['l'][self.num_full: self.num_full_v]
        var['lt'] = var['l'][self.num_full_v:]

        var['vf'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED).astype(np.float32)
        var['tf'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED).astype(np.float32)
        var['B'] = np.sign(self.alpha_v * var['v'] + self.alpha_t * var['t'] + self.eta * var['l'])

        # Iterations
        for epoch in range(self.Epoch):
            results = {}
            results['loss_labNet'] = []
            results['loss_imgNet'] = []
            results['loss_txtNet'] = []
            results['Loss_D'] = []
            results['mapl2l'] = []
            results['mapi2i'] = []
            results['mapt2t'] = []

            # adjust lr
            lr_lnet = self.lr_lab[epoch + 5]
            lr_inet = self.lr_img[epoch]
            lr_tnet = self.lr_txt[epoch]
            adjust_learning_rate(self.lnet_opt, lr_lnet)
            adjust_learning_rate(self.inet_opt, lr_inet)
            adjust_learning_rate(self.tnet_opt, lr_tnet)

            if epoch == 0:
                for k in range(15):
                    # train lnet
                    lr_lnet = self.lr_lab[k // 3]
                    adjust_learning_rate(self.lnet_opt, lr_lnet)
                    print('++++++++Start Training lnet++++++++')
                    train_labNet_loss = self.train_lab_net(var)
                    print('lab_net loss_total: %d' % train_labNet_loss)

            for k in range(5):
                # train lnet
                print('++++++++Start Training lnet++++++++')
                train_labNet_loss = self.train_lab_net(var)
                print('lab_net loss_total: %d' % train_labNet_loss)
                # train tnet
                print('++++++++Start Training tnet++++++++')
                train_txtNet_loss = self.train_txt_net(var)
                print('txt_net loss_total: %d' % train_txtNet_loss)
                # train inet
                print('++++++++Start Training inet++++++++')
                train_imgNet_loss = self.train_img_net(var)
                print('img_net loss_total: %d' % train_imgNet_loss)
                var['B'] = np.sign(var['v'] + var['t'] + var['l'])

            # test
            self.tnet.eval()
            self.inet.eval()
            with torch.no_grad():
                qBX, qBY = self.generate_code(self.loaders['qloader'])
                rBX = self.generate_code_single(self.loaders['vrloader'], 'image')
                rBY = self.generate_code_single(self.loaders['trloader'], 'text')
                mapi2t = calc_map(qBX, rBY, self.orig_data['L']['query'], self.orig_data['L']['retrieval_t'])
                mapt2i = calc_map(qBY, rBX, self.orig_data['L']['query'], self.orig_data['L']['retrieval_v'])
                mapi2i = calc_map(qBX, rBX, self.orig_data['L']['query'], self.orig_data['L']['retrieval_v'])
                mapt2t = calc_map(qBY, rBY, self.orig_data['L']['query'], self.orig_data['L']['retrieval_t'])

                condition_dir = './result-full-%f-lost-%f-%d' % (self.num_full, self.num_lost, self.bit)
                if not os.path.exists(condition_dir):
                    os.mkdir(condition_dir)

                save_dir_name = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
                cur_dir_path = os.path.join(condition_dir, save_dir_name)
                os.mkdir(cur_dir_path)

                savemat(os.path.join(cur_dir_path, 'B_all.mat'), {
                    'BxTest': qBX,
                    'BxTrain': rBX,
                    'ByTest': qBY,
                    'ByTrain': rBY,
                    'LQuery': self.orig_data['L']['query'],
                    'LxDB': self.orig_data['L']['retrieval_v'],
                    'LyDB': self.orig_data['L']['retrieval_t'],
                })

                with open(os.path.join(cur_dir_path, 'map.txt'), 'a') as f:
                    f.write('torch SEED: %d / cuda: %d\n' % (torch.initial_seed(), torch.cuda.initial_seed()))
                    f.write('==================================================\n')
                    f.write('...test map: map(i->t): %3.3f, map(t->i): %3.3f\n' % (mapi2t, mapt2i))
                    f.write('...test map: map(t->t): %3.3f, map(i->i): %3.3f\n' % (mapt2t, mapi2i))
                    f.write('==================================================\n')

                # save checkpoint
                state = {
                    'lnet': self.lnet.state_dict(),
                    'inet': self.inet.state_dict(),
                    'tnet': self.tnet.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, os.path.join(cur_dir_path, 'checkpoint'))
            self.tnet.train()
            self.inet.train()

    def train_lab_net(self, var):
        print('update label_net')
        L = var['l']
        V = var['v']
        T = var['t']
        B = var['B']
        loss_total = 0.0
        for batch in tqdm(self.loaders['lloader']):
            ind, label = batch

            label_feed = label.reshape([label.shape[0], 1, 1, label.shape[1]])
            hsh_l = self.lnet(label_feed.cuda())
            L[ind, :] = hsh_l.detach().cpu().numpy()

            # pairwise loss & quantization loss
            S = calc_neighbor(self.orig_data['L']['train'], label.numpy())
            S_cuda = torch.from_numpy(S).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()

            theta_FL = 1.0 / 4 * torch.from_numpy(V).cuda().mm(hsh_l.transpose(1, 0))
            loss_pair_hsh_FL = F.mse_loss(S_cuda.mul(theta_FL), F.softplus(theta_FL),
                                                      reduction='sum')
            theta_GL = 1.0 / 4 * torch.from_numpy(T).cuda().mm(hsh_l.transpose(1, 0))
            loss_pair_hsh_GL = F.mse_loss(S_cuda.mul(theta_GL), F.softplus(theta_GL),
                                                      reduction='sum')
            theta = 1.0 / 2 * torch.from_numpy(L).cuda().mm(hsh_l.transpose(1, 0))
            loss_pair_hsh = F.mse_loss(S_cuda.mul(theta), F.softplus(theta),
                                          reduction='sum')

            loss_quant_l = F.mse_loss(B_cuda, hsh_l, reduction='sum')
            loss_l = (loss_pair_hsh + loss_pair_hsh_FL + loss_pair_hsh_GL) \
                     + self.eta * loss_quant_l
            loss_total += float(loss_l.detach().cpu().numpy())

            self.lnet_opt.zero_grad()
            loss_l.backward()
            self.lnet_opt.step()
        return loss_total

    def incomplete_graph_contrastive_loss(self, hsh, M_c, M_ic, S_c, S_ic):
        M_all = torch.cat([torch.from_numpy(M_c).cuda(), torch.from_numpy(M_ic).cuda()], dim=0)
        S_all = torch.cat([S_c, S_ic], dim=0)
        theta_all = torch.sigmoid(1.0 / 2 * M_all.mm(hsh.transpose(1, 0)))
        loss = F.binary_cross_entropy_with_logits(theta_all.t(), S_all.t(), reduction='sum')
        return loss

    def calc_vague_sim(self, S):
        first = (((S.mm(S.t())) > 0.0) & (~torch.eye(S.shape[0], dtype=bool).cuda())).float()
        second = first.mm(S)
        S = (S + second > 0.0).float()
        return S

    def train_img_net(self, var):
        print('update image_net')
        V = var['v']
        T = var['t']
        L = var['l']
        B = var['B']
        VF = var['vf']
        TF = var['tf']
        loss_total = 0.0
        for batch in self.loaders['floader']:
            ind, image, text, label = batch

            fea_I, hsh_I, lab_I, fea_T_pred, mu_I, log_sigma_I = self.inet(image.cuda())
            V[ind, :] = hsh_I.detach().cpu().numpy()
            VF[ind, :] = fea_I.detach().cpu().numpy()

            # pairwise & quantization & classification losses
            S = calc_neighbor(self.orig_data['L']['train'], label.numpy())
            S_cuda = torch.from_numpy(S).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_MH = 1.0 / 2 * torch.from_numpy(L).cuda().mm(hsh_I.transpose(1, 0))
            Loss_pair_Hsh_MH = F.mse_loss(S_cuda.mul(theta_MH), F.softplus(theta_MH),
                                                      reduction='sum')
            ic_ind_select = np.random.randint(self.num_full, self.num_full_v, size=(self.ic_sel_num,))
            S_ic = self.calc_vague_sim(S_cuda[ic_ind_select, :])
            Loss_pair_Hsh_MO = self.incomplete_graph_contrastive_loss(
                hsh_I,
                np.concatenate([T[:self.num_full, :], T[self.num_full_v:, :]], axis=0),
                T[ic_ind_select, :],
                torch.cat([S_cuda[:self.num_full, :], S_cuda[self.num_full_v:, :]], dim=0),
                S_ic
            )
            Loss_quant_I = F.mse_loss(B_cuda, hsh_I, reduction='sum')
            # CVIB loss
            fea_T_real = torch.from_numpy(TF[ind, :]).cuda()
            Loss_prior_kl = torch.sum(mu_I.pow(2).add_(log_sigma_I.exp()).mul_(-1).add_(1).add_(log_sigma_I)).mul_(-0.5)
            Loss_cross_hash_MI = F.binary_cross_entropy_with_logits(fea_T_pred, torch.sigmoid(fea_T_real), reduction='sum') \
                                 + self.hyper_sigma * Loss_prior_kl
            loss_i = (Loss_pair_Hsh_MH + Loss_pair_Hsh_MO) + self.hyper_mi * Loss_cross_hash_MI\
                     + self.alpha_v * Loss_quant_I
            loss_total += float(loss_i.detach().cpu().numpy())

            self.inet_opt.zero_grad()
            loss_i.backward()
            self.inet_opt.step()

        for batch_v in tqdm(self.loaders['vloader']):
            ind, image, label = batch_v
            fea_I, hsh_I, lab_I, fea_T_pred, mu_I, log_sigma_I = self.inet(image.cuda())
            V[ind, :] = hsh_I.detach().cpu().numpy()
            VF[ind, :] = fea_I.detach().cpu().numpy()
            S = calc_neighbor(self.orig_data['L']['train'], label.numpy())
            S_cuda = torch.from_numpy(S).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_MH = 1.0 / 2 * torch.from_numpy(L).cuda().mm(hsh_I.transpose(1, 0))
            Loss_pair_Hsh_MH = F.mse_loss(S_cuda.mul(theta_MH), F.softplus(theta_MH),
                                          reduction='sum')
            ic_ind_select = np.random.randint(self.num_full, self.num_full_v, size=(self.ic_sel_num,))
            S_ic = self.calc_vague_sim(S_cuda[ic_ind_select, :])
            Loss_pair_Hsh_MO = self.incomplete_graph_contrastive_loss(
                hsh_I,
                np.concatenate([T[:self.num_full, :], T[self.num_full_v:, :]], axis=0),
                T[ic_ind_select, :],
                torch.cat([S_cuda[:self.num_full, :], S_cuda[self.num_full_v:, :]], axis=0),
                S_ic
            )
            Loss_quant_I = F.mse_loss(B_cuda, hsh_I, reduction='sum')
            loss_i = (Loss_pair_Hsh_MH + Loss_pair_Hsh_MO) \
                     + self.alpha_v * Loss_quant_I
            loss_total += float(loss_i.detach().cpu().numpy())

            self.inet_opt.zero_grad()
            loss_i.backward()
            self.inet_opt.step()

            # CVIB completion
            fea_T_completion = fea_T_pred.detach()
            with torch.no_grad():
                TF[ind, :] = fea_T_completion.cpu().numpy()
                T[ind, :] = self.tnet.get_hash(fea_T_completion).cpu().numpy()

        return loss_total

    def train_txt_net(self, var):
        print('update text_net')
        V = var['v']
        T = var['t']
        L = var['l']
        B = var['B']
        VF = var['vf']
        TF = var['tf']
        loss_total = 0.0
        for batch in self.loaders['floader']:
            ind, image, text, label = batch
            fea_T, hsh_T, lab_T, fea_I_pred, mu_T, log_sigma_T = self.tnet(text.cuda())
            T[ind, :] = hsh_T.detach().cpu().numpy()
            TF[ind, :] = fea_T.detach().cpu().numpy()
            S = calc_neighbor(self.orig_data['L']['train'], label.numpy())
            S_cuda = torch.from_numpy(S).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_MH = 1.0 / 2 * torch.from_numpy(L).cuda().mm(hsh_T.transpose(1, 0))
            Loss_pair_Hsh_MH = F.mse_loss(S_cuda.mul(theta_MH), F.softplus(theta_MH),
                                          reduction='sum')
            ic_ind_select = np.random.randint(self.num_full_v, self.num_train, size=(self.ic_sel_num,))
            S_ic = self.calc_vague_sim(S_cuda[ic_ind_select, :])
            Loss_pair_Hsh_MO = self.incomplete_graph_contrastive_loss(
                hsh_T,
                V[:self.num_full_v, :],
                V[ic_ind_select, :],
                S_cuda[:self.num_full_v, :],
                S_ic
            )
            Loss_quant_T = F.mse_loss(B_cuda, hsh_T, reduction='sum')
            # CVIB loss
            fea_I_real = torch.from_numpy(VF[ind, :]).cuda()
            Loss_prior_kl = torch.sum(mu_T.pow(2).add_(log_sigma_T.exp()).mul_(-1).add_(1).add_(log_sigma_T)).mul_(-0.5)
            Loss_cross_hash_MI = F.binary_cross_entropy_with_logits(fea_I_pred, torch.sigmoid(fea_I_real),
                                                                    reduction='sum') \
                                 + self.hyper_sigma * Loss_prior_kl
            loss_t = (Loss_pair_Hsh_MH + Loss_pair_Hsh_MO) + self.hyper_mi * Loss_cross_hash_MI \
                     + self.alpha_v * Loss_quant_T
            loss_total += float(loss_t.detach().cpu().numpy())

            self.tnet_opt.zero_grad()
            loss_t.backward()
            self.tnet_opt.step()

        for batch_t in tqdm(self.loaders['tloader']):
            ind, text, label = batch_t
            fea_T, hsh_T, lab_T, fea_I_pred, mu_T, log_sigma_T = self.tnet(text.cuda())
            T[ind, :] = hsh_T.detach().cpu().numpy()
            TF[ind, :] = fea_T.detach().cpu().numpy()
            S = calc_neighbor(self.orig_data['L']['train'], label.numpy())
            S_cuda = torch.from_numpy(S).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_MH = 1.0 / 2 * torch.from_numpy(L).cuda().mm(hsh_T.transpose(1, 0))
            Loss_pair_Hsh_MH = F.mse_loss(S_cuda.mul(theta_MH), F.softplus(theta_MH),
                                         reduction='sum')
            ic_ind_select = np.random.randint(self.num_full_v, self.num_train, size=(self.ic_sel_num,))
            S_ic = self.calc_vague_sim(S_cuda[ic_ind_select, :])
            Loss_pair_Hsh_MO = self.incomplete_graph_contrastive_loss(
                hsh_T,
                V[:self.num_full_v, :],
                V[ic_ind_select, :],
                S_cuda[:self.num_full_v, :],
                S_ic
            )
            Loss_quant_T = F.mse_loss(B_cuda, hsh_T, reduction='sum')
            loss_t =  (Loss_pair_Hsh_MH + Loss_pair_Hsh_MO) \
                     + self.alpha_v * Loss_quant_T
            loss_total += float(loss_t.detach().cpu().numpy())

            self.tnet_opt.zero_grad()
            loss_t.backward()
            self.tnet_opt.step()

            # CVIB completion
            fea_I_completion = fea_I_pred.detach()
            with torch.no_grad():
                VF[ind, :] = fea_I_completion.cpu().numpy()
                V[ind, :] = self.tnet.get_hash(fea_I_completion).cpu().numpy()
        return loss_total

    def generate_code(self, loader):
        num_data = len(loader.dataset)
        ind_shift = loader.dataset.ind_shift
        BX = np.zeros([num_data, self.bit], dtype=np.float32)
        BY = np.zeros([num_data, self.bit], dtype=np.float32)
        for batch in tqdm(loader):
            ind, image, text, label = batch
            ind = ind - ind_shift
            fea_I, hsh_I, lab_I, fea_T_pred, mu_I, log_sigma_I = self.inet(image.cuda())
            BX[ind, :] = hsh_I.cpu().numpy()
            fea_T, hsh_T, lab_T, fea_I_pred, mu_T, log_sigma_T = self.tnet(text.cuda())
            BY[ind, :] = hsh_T.cpu().numpy()
        BX = np.sign(BX)
        BY = np.sign(BY)
        return BX, BY

    def generate_code_single(self, loader, modal_name):
        num_data = len(loader.dataset)
        ind_shift = loader.dataset.ind_shift
        B = np.zeros([num_data, self.bit], dtype=np.float32)
        if modal_name == 'image':
            for batch in tqdm(loader):
                ind, image, label = batch
                ind = ind - ind_shift
                fea_I, hsh_I, lab_I, fea_T_pred, mu_I, log_sigma_I = self.inet(image.cuda())
                B[ind, :] = hsh_I.cpu().numpy()
        else:
            for batch in tqdm(loader):
                ind, text, label = batch
                ind = ind - ind_shift
                fea_T, hsh_T, lab_T, fea_I_pred, mu_T, log_sigma_T = self.tnet(text.cuda())
                B[ind, :] = hsh_T.cpu().numpy()
        B = np.sign(B)
        return B
