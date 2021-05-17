import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from tqdm import tqdm
import json

from torch.utils.tensorboard import SummaryWriter

from .model import Resnet
from .draw import *
from .dataset import XYbDataset
from .util import *


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


def train_block_corner_coord_model(model, xtr, ytr, xva, yva, epochs=100, start_epoch=1,
        batch_size=None, lr=(1e-3, 1e-2), save_interval=5, output_dir='outputs/', patience=10, device='cuda:0'):

    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'preds'), exist_ok=True)

    model.to(device)
    loss_func = nn.SmoothL1Loss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = len(xtr) / batch_size
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr[0], max_lr=lr[1],
        cycle_momentum=False, step_size_up=steps*4, step_size_down=steps*4)
    earlystop = EarlyStopping(patience=patience, path=os.path.join(output_dir, 'models/best.pt'), verbose=True)

    # log file header
    if start_epoch <= 1:
        write_file('epoch,tr_loss,va_loss',
                os.path.join(output_dir, 'training_log.txt'), mode='w')
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs/'))
    trdl = DataLoader(XYbDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    vadl = DataLoader(XYbDataset(xva, yva), batch_size=batch_size, shuffle=True)

    best_epoch = 0
    for epoch in range(start_epoch, start_epoch+epochs):
        print(f'\nepoch {epoch}')

        # ---------train---------
        model.train()
        trdl_bar = tqdm(trdl, ncols=100)
        tr_losses = []
        for batch, (xb, yb) in enumerate(trdl_bar):
            xb, yb = torch.tensor(xb).float().to(device), torch.tensor(yb).float().to(device)
            ypredb = model(xb)
            loss = loss_func(ypredb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss = loss.item()
            scheduler.step()

            tr_losses.append(tr_loss)
            trdl_bar.set_description(f'tr loss: {tr_loss:.3f}')
            writer.add_scalars('loss', {'tr': tr_loss} , epoch)

            if epoch % save_interval == 0 and xb is not None:
                with torch.no_grad():
                    path = os.path.join(output_dir, f'preds/e{epoch}/tr')
                    os.makedirs(path, exist_ok=True)
                    write_corners_xybs(
                        xb.cpu().numpy(), yb.cpu().numpy(), ypredb.cpu().numpy(),
                        output_dir=path, batch_no=batch, n_samples=5)

        # ---------validation---------
        model.eval()
        vadl_bar = tqdm(vadl, ncols=100)
        va_losses = []
        with torch.no_grad():
            for batch, (xb, yb) in enumerate(vadl_bar):
                xb, yb = torch.tensor(xb).float().to(device), torch.tensor(yb).float().to(device)
                ypredb = model(xb)
                loss = loss_func(ypredb, yb)

                va_loss = loss.item()
                va_losses.append(va_loss)
                vadl_bar.set_description(f'va loss: {va_loss:.3f}')
                writer.add_scalars("loss", {'va': va_loss}, epoch)

            if epoch % save_interval == 0 and xb is not None:
                path = os.path.join(output_dir, f'preds/e{epoch}/va/')
                os.makedirs(path, exist_ok=True)
                write_corners_xybs(
                    xb.cpu().numpy(), yb.cpu().numpy(), ypredb.cpu().numpy(),
                    output_dir=path, batch_no=batch, n_samples=5)
                torch.save(model, os.path.join(output_dir, f'models/{epoch}.pt'))

        tr_loss_mean = np.array(tr_losses).mean()
        va_loss_mean = np.array(va_losses).mean()
        print(f'tr loss: {tr_loss_mean}; va loss: {va_loss_mean}')
        lr = optimizer.param_groups[0]['lr']
        print(f'lr: {lr}')
        write_file(f'{epoch},{tr_loss_mean},{va_loss_mean}',
            os.path.join(output_dir, 'training_log.txt'), mode='a')

        earlystop(va_loss_mean, model)
        if earlystop.counter == 0:
            best_epoch = epoch
        if patience > 0 and earlystop.early_stop:
            break

    write_file(f'best epoch: {best_epoch}', os.path.join(output_dir, 'training_log.txt'), mode='a')
    writer.flush()


def finetune_block_corner_coord_model(model, xs, ys, epochs=100, start_epoch=1,
        batch_size=None, save_interval=0, output_dir='outputs/', device='cuda:0'):

    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'preds'), exist_ok=True)

    loss_func = nn.SmoothL1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2,
        cycle_momentum=False, step_size_up=100, step_size_down=100)

    # log file header
    if start_epoch <= 1:
        write_file('epoch,tr_loss',
                os.path.join(output_dir, 'training_log.txt'), mode='w')
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs/'))

    dl = DataLoader(XYbDataset(xs, ys), batch_size=batch_size, shuffle=True)
    for epoch in range(start_epoch, start_epoch+epochs+1):
        print(f'\nepoch {epoch}')

        # ---------train---------
        model.train()
        trdl_bar = tqdm(dl, ncols=100)
        tr_losses, xbs, ybs, ypredbs = [], [], [], []
        for xb, yb in trdl_bar:
            if xb is None:
                print("empty batch"); continue
            xb, yb = torch.tensor(xb).float().to(device), torch.tensor(yb).float().to(device)
            ypredb = model(xb)
            loss = loss_func(ypredb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss = loss.item()
            tr_losses.append(tr_loss)
            trdl_bar.set_description(f'tr loss: {tr_loss:.3f}')
            writer.add_scalars("loss", {'tr': tr_loss} , epoch)

            with torch.no_grad():
                xbs.append(xb.cpu().numpy())
                ybs.append(yb.cpu().numpy())
                ypredbs.append(ypredb.cpu().numpy())

            scheduler.step()

        if epoch % save_interval == 0 and xbs is not None:
            path = os.path.join(output_dir, f'preds/e{epoch}/tr')
            os.makedirs(path, exist_ok=True)
            write_corners_xybs(xbs, ybs, ypredbs, output_dir=path, n_samples=5)
            torch.save(model, os.path.join(output_dir, f'models/{epoch}.pt'))

        tr_loss_mean = np.array(tr_losses).mean()
        lr = optimizer.param_groups[0]['lr']
        print(f'lr: {lr}')
        print(f'tr loss: {tr_loss_mean}')
        write_file(f'{epoch},{tr_loss_mean}',
            os.path.join(output_dir, 'training_log.txt'), mode='a')
    writer.flush()


def train_spot_coord_model(data_dirs, gal_dirs, va_size=.2, te_size=.2, down_sample=4,
    window_expand=2, equalize=False, morphology=False, model=None, epochs=100, start_epoch=0,
    chip_batch_size=4, batch_size=None, save_interval=5, output_dir='outputs/', device='cuda:0'):

    # os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'preds'), exist_ok=True)

    dataset = SpotCoordDataset(
        window_expand=window_expand,
        down_sample=down_sample,
        equalize=equalize, morphology=morphology)
    trdl, vadl, tedl = dataset.get_dataloaders(data_dirs, gal_dirs, batch_size=chip_batch_size,
        va_size=va_size, te_size=te_size, save_dir=output_dir)
    print(len(trdl.dataset), len(vadl.dataset), len(tedl.dataset))

    # model = Resnet().to(device) if model == None else model.to(device)
    # loss_func = nn.MSELoss(reduction='mean')
    # # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # # log file header
    # write_file('epoch,tr_loss,va_loss',
    #         os.path.join(output_dir, 'training_log.txt'), mode='w')

    for epoch in range(start_epoch, start_epoch+epochs):
        print(f'\nepoch {epoch}')

        trdl_bar = tqdm(trdl, ncols=100)
        tr_losses = []
        for ims, gals, gprs in trdl_bar:
            xb, yb, _ = dataset.imgs2xy(ims, gals, gprs, augment=True)
            draw_xy_spot_coord(xb, yb); exit()
