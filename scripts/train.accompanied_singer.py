#!/usr/bin/env python

import os
gid = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)

import time
import pickle
import random
from collections import OrderedDict

from training_manager import TrainingManager, get_current_time

import numpy as np

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

torch.multiprocessing.set_sharing_strategy('file_system')


class VocKeyboardPitchDataset(Dataset):
    def __init__(self, ids, dir_vocal, dir_pitch):
        self.metadata = ids
        self.dir_vocal = dir_vocal
        self.dir_pitch = dir_pitch

    def __getitem__(self, index):
        id = self.metadata[index]
        voc_fp = os.path.join(self.dir_vocal, id, 'vocals.npy')
        kpitch_fp = os.path.join(self.dir_pitch, id, 'keyboard.88pitches.npy')

        voc = np.load(voc_fp)
        kpitch = np.load(kpitch_fp)

        return voc, kpitch

    def __len__(self):
        return len(self.metadata)


def get_vockeyboardpitch_datasets(path, feat_type, batch_size, va_samples):

    dataset_fp = os.path.join(path, 'dataset.pkl')
    in_dir = os.path.join(path, feat_type)
    with open(dataset_fp, 'rb') as f:
        dataset = pickle.load(f)

    pitch_dir = os.path.join(path, 'transcription')

    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)

    va_ids = dataset_ids[-va_samples:]
    tr_ids = dataset_ids[:-va_samples]

    tr_dataset = VocKeyboardPitchDataset(tr_ids, in_dir, pitch_dir)
    va_dataset = VocKeyboardPitchDataset(va_ids, in_dir, pitch_dir)
    num_tr = len(tr_dataset)
    num_va = len(va_dataset)

    iterator_tr = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    iterator_va = DataLoader(
        va_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=True,
        pin_memory=True)

    return iterator_tr, num_tr, iterator_va, num_va


def validate():
    # Store random state
    cpu_rng_state_tr = torch.get_rng_state()
    gpu_rng_state_tr = torch.cuda.get_rng_state()

    # Set random stae
    torch.manual_seed(123)

    # ###
    sum_losses_va = OrderedDict([(loss_name, 0) for loss_name in loss_funcs])

    count_all_va = 0

    # In validation, set netG.eval()
    netG.eval()
    netVD.eval()
    num_batches_va = len(iterator_va)
    with torch.set_grad_enabled(False):
        for i_batch, batch in enumerate(iterator_va):

            # voc.shape=(bs, feat_dim, num_frames)
            voc, kpitch = batch
            voc = voc[:, :, :kpitch.size(2)]

            voc = voc.cuda()
            kpitch = kpitch.float().cuda()

            # ### Make pitch condition ###
            pitch_cond = kpitch

            # ### Prepare targets
            bs, _, nf = voc.size()

            # ### Train generator ###
            z = torch.zeros((bs, z_dim, nf)).normal_(0, 1).float().cuda()
            zp = torch.cat([z, pitch_cond], dim=1)

            gen_voc = netG(zp)

            gloss_v = torch.mean(torch.abs(netVD(gen_voc, pitch_cond)-gen_voc))

            # ### Train discriminators ###
            real_dloss_v = torch.mean(torch.abs(netVD(voc, pitch_cond)-voc))
            fake_dloss_v = torch.mean(torch.abs(netVD(gen_voc.detach(), pitch_cond)-gen_voc.detach()))

            dloss_v = real_dloss_v - k_v * fake_dloss_v

            # ### Convergence ###
            diff_v = torch.mean(gamma * real_dloss_v - fake_dloss_v)

            convergence_v = (real_dloss_v + torch.abs(diff_v))

            # ### Losses ###
            losses = OrderedDict([
                ('VG', gloss_v),
                ('VD', dloss_v),
                ('RealVD', real_dloss_v),
                ('FakeVD', fake_dloss_v),
                ('ConvergenceV', convergence_v),
            ])

            # ### Misc ###
            count_all_va += bs

            # Accumulate losses
            losses_va = OrderedDict([(loss_name, lo.item()) for loss_name, lo in losses.items()])

            for loss_name, lo in losses_va.items():
                sum_losses_va[loss_name] += lo*bs

            if i_batch % 10 == 0:
                print('{}/{}'.format(i_batch, num_batches_va))
    mean_losses_va = OrderedDict([(loss_name, slo / count_all_va) for loss_name, slo in sum_losses_va.items()])

    # Restore rng state
    torch.set_rng_state(cpu_rng_state_tr)
    torch.cuda.set_rng_state(gpu_rng_state_tr)

    return mean_losses_va


def make_inf_iterator(data_iterator):
    while True:
        for data in data_iterator:
            yield data


class G3Block(nn.Module):
    def __init__(self, feat_dim, ks, dilation, groups):
        super().__init__()
        ksm1 = ks-1
        mfd = feat_dim
        di = dilation
        self.g = groups

        self.relu = nn.LeakyReLU()

        self.rec = nn.GRU(mfd, mfd, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(mfd, mfd, ks, 1, ksm1*di//2, dilation=di, groups=groups)
        self.gn = nn.GroupNorm(groups, mfd)

    def forward(self, x):
        bs, mfd, nf = x.size()

        r = x.transpose(1, 2)
        r, _ = self.rec(r)
        r = r.transpose(1, 2).view(bs, 2, mfd, nf).sum(1)
        c = self.relu(self.gn(self.conv(r)))
        x = x+r+c

        return x


class NetG(nn.Module):
    def __init__(self, feat_dim, z_dim, freq_dim):
        super().__init__()

        ks = 3  # filter size
        mfd = 512
        groups = 4
        self.groups = groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.freq_dim = freq_dim

        # ### Main body ###
        blocks = [
            nn.Conv1d(z_dim+freq_dim, mfd, 3, 1, 1),
            nn.GroupNorm(groups, mfd),
            nn.LeakyReLU(),
            G3Block(mfd, ks, dilation=2, groups=groups),
            G3Block(mfd, ks, dilation=4, groups=groups),
        ]
        self.body = nn.Sequential(*blocks)

        # ### Head ###
        self.head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

    def forward(self, z):

        # Body
        x = self.body(z)

        # Head
        # shape=(bs, feat_dim, nf)
        x = torch.sigmoid(self.head(x))

        return x


class NetVD(nn.Module):
    def __init__(self, input_size, freq_dim):
        super().__init__()

        ks = 3  # filter size
        mfd = 512
        groups = 4
        self.groups = groups
        self.mfd = mfd

        self.input_size = input_size
        self.freq_dim = freq_dim

        # ### Main body ###
        blocks = [
            nn.Conv1d(input_size+freq_dim, mfd, 3, 1, 1),
            nn.GroupNorm(groups, mfd),
            nn.LeakyReLU(),
            G3Block(mfd, ks, dilation=2, groups=groups),
            G3Block(mfd, ks, dilation=4, groups=groups),
        ]
        self.body = nn.Sequential(*blocks)

        self.head = nn.Conv1d(mfd, input_size, 3, 1, 1)

    def forward(self, x, cond):
        bs, fd, nf = x.size()

        # Body
        x = self.body(torch.cat([x, cond], dim=1))

        # Head
        # shape=(bs, input_size, nf)
        out = torch.sigmoid(self.head(x))

        return out


if __name__ == '__main__':
    model_id = None
    resume_metric = 'ConvergenceV'

    if model_id is None:
        model_id = get_current_time()
        resume_training = False
    else:
        resume_training = True

    script_path = os.path.realpath(__file__)
    print(model_id)
    print(script_path)

    # Options
    base_dir = ""
    base_out_dir = base_dir

    data_dir = ""

    feat_dim = 80
    z_dim = 20
    freq_dim = 88

    num_va = 200

    feat_type = 'mel'

    loss_funcs = OrderedDict([
        ('VG', None),
        ('VD', None),
        ('RealVD', None),
        ('FakeVD', None),
        ('ConvergenceV', None),
    ])

    # BEGAN parameters
    gamma = 1.0
    lambda_k = 0.01
    k_v = 0.0

    # #############################################################
    # ### Set the validation losses that are used in evaluation ###
    # #############################################################
    eval_metrics = [
        ('ConvergenceV', 'lower_better'),
    ]
    # #############################################################

    # Training options
    init_lr = 0.0001
    num_epochs = 500
    batches_per_epoch = 500

    max_grad_norm = 3

    save_rate = 10

    batch_size = 5

    # Dirs and fps
    save_dir = os.path.join(base_out_dir, 'save.accompaniment_conditioned')
    output_dir = os.path.join(save_dir, model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iterator_tr, num_tr, iterator_va, _ = get_vockeyboardpitch_datasets(
        data_dir, feat_type, batch_size, num_va)
    print('tr: {}, va: {}'.format(num_tr, num_va))

    inf_iterator_tr = make_inf_iterator(iterator_tr)

    # Model
    netG = NetG(feat_dim, z_dim, freq_dim).cuda()
    netVD = NetVD(feat_dim, freq_dim).cuda()

    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=init_lr)
    optimizerVD = optim.Adam(netVD.parameters(), lr=init_lr)

    # ###################################
    # ### Initialize training manager ###
    # ###################################
    manager = TrainingManager(
        [netG, netVD],  # networks
        [optimizerG, optimizerVD],  # optimizers, could be None
        ['Singer', 'VocD'],  # names of the corresponding networks
        output_dir, save_rate, script_path=script_path)
    # ###################################

    # ### Resume training ###
    if resume_training:
        init_epoch = manager.resume_training(model_id, save_dir, resume_metric)
    else:
        init_epoch = 1
        manager.save_initial()

    # ### Train ###
    for epoch in range(init_epoch, 1+num_epochs):
        print(model_id)
        t0 = time.time()

        # ### Training ###
        print('Training...')
        sum_losses_tr = OrderedDict([(loss_name, 0) for loss_name in loss_funcs])

        count_all_tr = 0

        # num_batches_tr = len(iterator_tr)
        num_batches_tr = batches_per_epoch

        tt0 = time.time()

        # In training, set net.train()
        netG.train()
        netVD.train()
        for i_batch in range(batches_per_epoch):
            batch = next(inf_iterator_tr)

            # voc.shape=(bs, feat_dim, num_frames)
            voc, kpitch = batch
            voc = voc[:, :, :kpitch.size(2)]

            voc = voc.cuda()
            kpitch = kpitch.float().cuda()

            # ### Make pitch condition ###
            pitch_cond = kpitch

            # ### Prepare targets
            bs, _, nf = voc.size()

            # ### Train generator ###
            netG.zero_grad()

            z = torch.zeros((bs, z_dim, nf)).normal_(0, 1).float().cuda()
            zp = torch.cat([z, pitch_cond], dim=1)

            gen_voc = netG(zp)

            gloss_v = torch.mean(torch.abs(netVD(gen_voc, pitch_cond)-gen_voc))

            # Back propagation
            gloss_v.backward(retain_graph=True)
            if max_grad_norm is not None:
                clip_grad_norm_(netG.parameters(), max_grad_norm)

            optimizerG.step()

            # ### Train discriminators ###
            netVD.zero_grad()

            real_dloss_v = torch.mean(torch.abs(netVD(voc, pitch_cond)-voc))
            fake_dloss_v = torch.mean(torch.abs(netVD(gen_voc.detach(), pitch_cond)-gen_voc.detach()))

            dloss_v = real_dloss_v - k_v * fake_dloss_v

            # Update
            dloss_v.backward(retain_graph=True)
            if max_grad_norm is not None:
                clip_grad_norm_(netVD.parameters(), max_grad_norm)

            optimizerVD.step()

            # ### Convergence ###
            diff_v = torch.mean(gamma * real_dloss_v - fake_dloss_v)

            k_v = k_v + lambda_k * diff_v.item()

            k_v = min(max(k_v, 0), 1)

            convergence_v = (real_dloss_v + torch.abs(diff_v))

            # ### Losses ###
            losses = OrderedDict([
                ('VG', gloss_v),
                ('VD', dloss_v),
                ('RealVD', real_dloss_v),
                ('FakeVD', fake_dloss_v),
                ('ConvergenceV', convergence_v),
            ])

            # ### Misc ###

            # Accumulate losses
            losses_tr = OrderedDict([(loss_name, lo.item()) for loss_name, lo in losses.items()])

            for loss_name, lo in losses_tr.items():
                sum_losses_tr[loss_name] += lo

            count_all_tr += 1

            # ### Print info ###
            if i_batch % 10 == 0:
                batch_info = 'Epoch {}. Batch: {}/{}, T: {:.3f}, '.format(epoch, i_batch, num_batches_tr, time.time()-tt0) + \
                    ''.join(['(tr) {}: {:.5f}, '.format(loss_name, lo) for loss_name, lo in losses_tr.items()])
                print(batch_info, k_v)
            tt0 = time.time()

        # Compute average loss
        mean_losses_tr = OrderedDict([(loss_name, slo / count_all_tr) for loss_name, slo in sum_losses_tr.items()])

        # ### Validation ###
        print('')
        print('Validation...')
        mean_losses_va = validate()

        # ###########################################################################
        # ### Check the best validation losses and save information in the middle ###
        # ###########################################################################

        # Check and update the best validation loss
        va_metrics = [
            (metric_name, mean_losses_va[metric_name], higher_or_lower)
            for metric_name, higher_or_lower in eval_metrics
        ]
        best_va_metrics = manager.check_best_va_metrics(va_metrics, epoch)

        # Save the record
        record = {
            'mean_losses_tr': mean_losses_tr,
            'mean_losses_va': mean_losses_va,
            'best_va_metrics': best_va_metrics,
        }

        manager.save_middle(epoch, record, va_metrics)
        manager.print_record(record)
        manager.print_record_in_one_line(best_va_metrics)
        # ###########################################################################

        t1 = time.time()
        print('Epoch: {} finished. Time: {:.3f}. Model ID: {}'.format(epoch, t1-t0, model_id))

    # #############################
    # ### Save the final record ###
    # #############################
    manager.save_final()
    # #############################

    print(model_id)
