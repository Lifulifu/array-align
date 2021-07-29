import cv2
import os
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import learn2learn as l2l

from .model import Resnet
from .draw import *
from .dataset import *
from .util import *
from .train import EarlyStopping


class Reptile():
    def __init__(self, model, tr_tasks=[], te_tasks=[], epsilon=1, channels=1,
                 meta_start_epoch=1, meta_epochs=1000, meta_batch_size=4, augment=5, patience=0,
                 meta_lr=1e-3, task_epochs=10, task_tr_size=5, task_te_size=None, task_te_epochs=100,
                 task_lr=1e-3, save_interval=5, output_dir='outputs/', device='cuda:0'):
        self.model = model.to(device)
        self.device = device
        self.tr_tasks = tr_tasks
        self.te_tasks = te_tasks
        self.epsilon = epsilon
        self.channels = channels

        self.meta_start_epoch = meta_start_epoch
        self.meta_epochs = meta_epochs
        self.meta_lr = meta_lr
        self.meta_batch_size = meta_batch_size
        self.patience = patience

        self.task_epochs = task_epochs
        self.task_te_epochs = task_te_epochs
        self.task_tr_size = task_tr_size
        self.task_te_size = task_te_size
        self.task_lr = task_lr
        self.augment = augment

        self.save_interval = save_interval
        self.output_dir = output_dir

        self.aug_seq = aug.Sequential([
            # aug.HorizontalFlip(0.5),
            # aug.VerticalFlip(0.5),
            aug.Affine(
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-5, 5),
                mode='wrap'
            )], random_order=True)

    def average_states(self, dict_list):
        """
        Averrage of list of state_dicts.
        """
        for param_tensor in dict_list[0]:
            for i in range(1, len(dict_list)):
                dict_list[0][param_tensor] = dict_list[0][param_tensor] + \
                    dict_list[i][param_tensor]
            dict_list[0][param_tensor] = dict_list[0][param_tensor] / \
                len(dict_list)
        average_var = dict_list[0]
        return average_var

    def interpolate_states(self, old_vars, new_vars, epsilon):
        """
        Interpolate between two sequences of variables.
        """
        for param_tensor in new_vars:
            new_vars[param_tensor] = old_vars[param_tensor] + \
                (new_vars[param_tensor] - old_vars[param_tensor]) * epsilon
        return new_vars

    def train_task(self, xtr, xte, ytr, yte):
        '''
        will not update self.model
        '''
        print(xtr.shape, xte.shape, ytr.shape, yte.shape)
        new_model = Resnet(channels=self.channels).to(self.device)
        new_model.load_state_dict(self.model.state_dict())  # copy model
        xtr, xte, ytr, yte = torch.tensor(xtr).float().to(self.device), torch.tensor(xte).float().to(
            self.device), torch.tensor(ytr).float().to(self.device), torch.tensor(yte).float().to(self.device)
        # optimizer = torch.optim.SGD(new_model.parameters(), lr=self.task_lr)
        optimizer = torch.optim.AdamW(new_model.parameters(), lr=self.task_lr)
        loss_func = nn.SmoothL1Loss(reduction='mean')

        # train
        bar = tqdm(range(self.task_epochs))
        for i in bar:
            ypred = new_model(xtr)
            tr_loss = loss_func(ypred, ytr)
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
            bar.set_description(f'task tr loss: {tr_loss.item():.3f}')

        # test
        with torch.no_grad():
            ypred = new_model(xte)
            te_loss = loss_func(ypred, yte)

        return new_model, te_loss.item(), ypred.cpu().numpy()  # for xte

    def batch_train_update(self, epoch):
        '''
        will update self.model by load_state_dict
        '''
        states = []
        task_losses = []
        for b in range(self.meta_batch_size):
            task = np.random.choice(self.tr_tasks)
            x, y = np.load(os.path.join(task, 'x.npy')), np.load(
                os.path.join(task, 'y.npy'))
            if len(x) <= self.task_tr_size:
                continue
            print(x.shape, y.shape)
            xtr, xte, ytr, yte = self.split_trte(x, y)
            if self.augment > 0:
                xtr_aug, ytr_aug = self.im_augment(xtr, ytr, self.augment)
                xtr, ytr = np.concatenate((xtr, xtr_aug)), np.concatenate((ytr, ytr_aug))
            new_model, task_loss, ypred = self.train_task(xtr, xte, ytr, yte)
            task_losses.append(task_loss)
            states.append(new_model.state_dict())

        if epoch % self.save_interval == 0 and xte is not None:
            # save img results of the last task in this batch
            path = os.path.join(self.output_dir, f'preds/e{epoch}/tr/')
            os.makedirs(path, exist_ok=True)
            write_corners_xybs(xte, yte, ypred, output_dir=path, n_samples=0)

        # update meta model
        states = self.average_states(states)
        states = self.interpolate_states(
            self.model.state_dict(), states, self.epsilon)
        self.model.load_state_dict(states)

        return np.array(task_losses).mean()

    def im_augment(self, xs, ys, amount):
        augxs, augys = [], []
        while len(augxs) < amount:
            i = np.random.randint(len(xs))
            x, y = xs[i], ys[i]
            augx, coord = self.run_aug(x, y)
            if augx is None:
                continue
            augxs.append(augx)
            augys.append(coord.flatten())
        return augxs, augys

    def run_aug(self, img, y):
        '''
        img: (2, h, w)
        y: (3, 2)
        '''
        kpts = self.to_kpts(y, img.shape[1:])
        img = np.moveaxis(img*255, 0, -1).astype('uint8')
        img_aug, kpts_aug = self.aug_seq(image=img, keypoints=kpts)  # img: (2, w, h) -> (w, h, 2)
        coord = kpts_aug.to_xy_array()  # (3, 2)
        if self.check_out_of_bounds(img_aug.shape[:2], coord):
            return None, None  # skip if coord out of bounds
        # take only r and g channels
        return np.moveaxis(img_aug/255, -1, 0), coord

    def to_kpts(self, coords, shape):
        return KeypointsOnImage([
            Keypoint(x=coord[0], y=coord[1]) for coord in coords.reshape(3, 2)
        ], shape=shape)

    def check_out_of_bounds(self, img_shape, coord):
        h, w = img_shape
        return (coord[:, 0].max() >= w) or (coord[:, 1].max() >= h) or (
                coord[:, 0].min() < 0) or (coord[:, 1].min() < 0)

    def meta_train(self):
        '''
        data_dirs (list): should have file structure:
            dir/
            0/
                x.npy
                y.npy
            1/
                x.npy
                y.npy
            ...
        '''
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'preds'), exist_ok=True)

        # log file header
        if self.meta_start_epoch <= 1:
            write_file('epoch,tr_loss,va_loss',
                       os.path.join(self.output_dir, 'training_log.txt'), mode='w')
        writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'logs/'))
        earlystop = EarlyStopping(patience=self.patience, path=os.path.join(self.output_dir, 'models/best.pt'), verbose=True)

        for epoch in range(self.meta_start_epoch, self.meta_start_epoch + self.meta_epochs):
            # --- Train ---
            print(f'epoch {epoch}')
            self.model.train()
            meta_tr_loss = self.batch_train_update(epoch)

            # --- Validate ---
            print(f'validate')
            self.model.eval()
            task = np.random.choice(self.te_tasks)
            x, y = np.load(os.path.join(task, 'x.npy')), np.load(
                os.path.join(task, 'y.npy'))
            if len(x) <= self.task_tr_size:
                continue
            xtr, xte, ytr, yte = self.split_trte(x, y)
            if self.augment > 0:
                xtr_aug, ytr_aug = self.im_augment(xtr, ytr, self.augment)
                xtr, ytr = np.concatenate((xtr, xtr_aug)), np.concatenate((ytr, ytr_aug))
            _, meta_va_loss, ypred = self.train_task(xtr, xte, ytr, yte)

            writer.add_scalars("tr_loss", {
                'tr': meta_tr_loss,
                'va': meta_va_loss}, epoch)
            print(
                f'meta tr loss: {meta_tr_loss:.3f}, meta va loss: {meta_va_loss:.3f}')

            if epoch % self.save_interval == 0 and xte is not None:
                path = os.path.join(self.output_dir, f'preds/e{epoch}/va/')
                os.makedirs(path, exist_ok=True)
                write_corners_xybs(xte, yte, ypred, output_dir=path, n_samples=10)
                write_file(task, os.path.join(path, f'task.txt'))
                torch.save(self.model, os.path.join(
                    self.output_dir, f'models/{epoch}.pt'))

            write_file(f'{epoch},{meta_tr_loss},{meta_va_loss}',
                       os.path.join(self.output_dir, 'training_log.txt'), mode='a')
        writer.flush()

    def split_trte(self, x, y):
        idx = list(range(len(x)))
        np.random.shuffle(idx)
        x, y = x[idx], y[idx]
        xtr = x[:self.task_tr_size]
        xte = x[self.task_tr_size: self.task_tr_size +
                self.task_te_size] if self.task_te_size else x[self.task_tr_size:]
        ytr = y[:self.task_tr_size]
        yte = y[self.task_tr_size: self.task_tr_size +
                self.task_te_size] if self.task_te_size else y[self.task_tr_size:]
        return xtr, xte, ytr, yte


class MAML():
    def __init__(self, model, tr_tasks, te_tasks, epsilon=1, channels=1,
                 meta_start_epoch=1, meta_epochs=1000, meta_batch_size=4, augment=5,
                 meta_lr=1e-3, task_epochs=10, task_tr_size=5, task_te_size=None, task_te_epochs=100,
                 task_lr=1e-3, save_interval=5, output_dir='outputs/', device='cuda:0'):
        self.model = model.to(device)
        self.maml = l2l.algorithms.MAML(model, lr=task_lr, first_order=True, allow_unused=True)
        self.optimizer = torch.optim.Adam(self.maml.parameters(), meta_lr)
        self.loss_func = nn.SmoothL1Loss(reduction='mean')

        self.device = device
        self.tr_tasks = tr_tasks
        self.te_tasks = te_tasks
        self.channels = channels
        self.meta_start_epoch = meta_start_epoch
        self.meta_epochs = meta_epochs
        self.meta_lr = meta_lr
        self.meta_batch_size = meta_batch_size

        self.task_epochs = task_epochs
        self.task_te_epochs = task_te_epochs
        self.task_tr_size = task_tr_size
        self.task_te_size = task_te_size
        self.task_lr = task_lr
        self.augment = augment

        self.save_interval = save_interval
        self.output_dir = output_dir

    def train_task(self, xtr, xte, ytr, yte):
        '''
        will not update self.model
        '''
        print(xtr.shape, xte.shape, ytr.shape, yte.shape)
        xtr, xte, ytr, yte = torch.tensor(xtr).float().to(self.device), torch.tensor(xte).float().to(
            self.device), torch.tensor(ytr).float().to(self.device), torch.tensor(yte).float().to(self.device)
        learner = self.maml.clone()
        bar = tqdm(range(self.task_epochs))
        for i in bar: # adaptation_steps
            ypredtr = learner(xtr)
            tr_loss = self.loss_func(ypredtr, ytr)
            learner.adapt(tr_loss)
            bar.set_description(f'task tr loss: {tr_loss.item():.3f}')

        ypredte = learner(xte)
        te_loss = self.loss_func(ypredte, yte)

        return te_loss, ypredte.detach().cpu().numpy()

    def batch_train_update(self):
        '''
        will update self.model by load_state_dict
        '''
        meta_train_loss = 0.
        for b in range(self.meta_batch_size):
            # Sample task
            task = np.random.choice(self.tr_tasks)
            x, y = np.load(os.path.join(task, 'x.npy')), np.load(
                os.path.join(task, 'y.npy'))
            if len(x) <= self.task_tr_size:
                continue
            print(x.shape, y.shape)
            xtr, xte, ytr, yte = self.split_trte(x, y)
            task_loss, _ = self.train_task(xtr, xte, ytr, yte)
            meta_train_loss += task_loss

        meta_train_loss /= self.meta_batch_size
        self.optimizer.zero_grad()
        meta_train_loss.backward()
        self.optimizer.step()

        return meta_train_loss.item()

    def meta_train(self):
        '''
        data_dirs (list): should have file structure:
            dir/
            0/
                x.npy
                y.npy
            1/
                x.npy
                y.npy
            ...
        '''
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'preds'), exist_ok=True)

        # log file header
        if self.meta_start_epoch <= 1:
            write_file('epoch,tr_loss,va_loss',
                       os.path.join(self.output_dir, 'training_log.txt'), mode='w')
        writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'logs/'))

        for epoch in range(self.meta_start_epoch, self.meta_start_epoch + self.meta_epochs):
            # --- Train ---
            print(f'epoch {epoch}')
            self.model.train()
            meta_tr_loss = self.batch_train_update()

            # --- Validate ---
            print(f'validate')
            self.model.eval()
            task = np.random.choice(self.te_tasks)
            x, y = np.load(os.path.join(task, 'x.npy')), np.load(
                os.path.join(task, 'y.npy'))
            if len(x) <= self.task_tr_size:
                continue
            xtr, xte, ytr, yte = self.split_trte(x, y)
            meta_va_loss, ypred = self.train_task(xtr, xte, ytr, yte)

            writer.add_scalars("tr_loss", {
                'tr': meta_tr_loss,
                'va': meta_va_loss}, epoch)
            print(
                f'meta tr loss: {meta_tr_loss:.3f}, meta va loss: {meta_va_loss:.3f}')

            if epoch % self.save_interval == 0 and xte is not None:
                path = os.path.join(self.output_dir, f'preds/e{epoch}/va/')
                os.makedirs(path, exist_ok=True)
                write_corners_xybs(xte, yte, ypred, output_dir=path, n_samples=10)
                write_file(task, os.path.join(path, f'task.txt'))
                torch.save(self.model, os.path.join(
                    self.output_dir, f'models/{epoch}.pt'))

            write_file(f'{epoch},{meta_tr_loss},{meta_va_loss}',
                       os.path.join(output_dir, 'training_log.txt'), mode='a')
        writer.flush()

    def split_trte(self, x, y):
        idx = list(range(len(x)))
        np.random.shuffle(idx)
        x, y = x[idx], y[idx]
        xtr = x[:self.task_tr_size]
        xte = x[self.task_tr_size: self.task_tr_size +
                self.task_te_size] if self.task_te_size else x[self.task_tr_size:]
        ytr = y[:self.task_tr_size]
        yte = y[self.task_tr_size: self.task_tr_size +
                self.task_te_size] if self.task_te_size else y[self.task_tr_size:]
        return xtr, xte, ytr, yte
