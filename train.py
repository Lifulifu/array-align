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

def train_block_corner_coord_model(model, dataset, trdl, vadl, augment=0, bsa_args=None, epochs=100, start_epoch=1,
        batch_size=None, save_interval=5, output_dir='outputs/', device='cuda:0'):

    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'preds'), exist_ok=True)

    loss_func = nn.SmoothL1Loss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2,
        cycle_momentum=False, step_size_up=100, step_size_down=100)

    # log file header
    if start_epoch <= 1:
        write_file('epoch,tr_loss,va_loss',
                os.path.join(output_dir, 'training_log.txt'), mode='w')
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs/'))

    for epoch in range(start_epoch, start_epoch+epochs):
        print(f'\nepoch {epoch}')

        # ---------train---------
        model.train()
        trdl_bar = tqdm(trdl, ncols=100)
        tr_losses, xbs, ybs, ypredbs = [], [], [], []
        for ims, gals, gprs in trdl_bar:
            xb, yb = dataset.imgs2xy(ims, gals, gprs, augment=augment, bsa_args=bsa_args)
            if xb is None:
                print("empty batch"); continue
            if batch_size:
                idx = np.random.randint(low=0, high=len(xb), size=batch_size)
                xb, yb = xb[idx], yb[idx]
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

        # ---------validation---------
        model.eval()
        vadl_bar = tqdm(vadl, ncols=100)
        va_losses, xbs, ybs, ypredbs = [], [], [], []
        with torch.no_grad():
            for b, (ims, gals, gprs) in enumerate(vadl_bar):
                xb, yb = dataset.imgs2xy(ims, gals, gprs, augment=0)  # No aug in va
                if xb is None:
                    print("empty batch")
                    continue
                if batch_size:
                    idx = np.random.randint(low=0, high=len(xb), size=batch_size)
                    xb, yb = xb[idx], yb[idx]
                xb, yb = torch.tensor(xb).float().to(device), torch.tensor(yb).float().to(device)
                ypredb = model(xb)
                loss = loss_func(ypredb, yb)
                va_loss = loss.item()
                va_losses.append(va_loss)
                vadl_bar.set_description(f'va loss: {va_loss:.3f}')
                writer.add_scalars("loss", {'va': va_loss}, epoch)

                xbs.append(xb.cpu().numpy())
                ybs.append(yb.cpu().numpy())
                ypredbs.append(ypredb.cpu().numpy())

        if epoch % save_interval == 0 and xbs is not None:
            path = os.path.join(output_dir, f'preds/e{epoch}/va/')
            os.makedirs(path, exist_ok=True)
            write_corners_xybs(xbs, ybs, ypredbs, output_dir=path, n_samples=5)
            torch.save(model, os.path.join(output_dir, f'models/{epoch}.pt'))

        tr_loss_mean = np.array(tr_losses).mean()
        va_loss_mean = np.array(va_losses).mean()
        print(f'tr loss: {tr_loss_mean}; va loss: {va_loss_mean}')
        lr = optimizer.param_groups[0]['lr']
        print(f'lr: {lr}')
        write_file(f'{epoch},{tr_loss_mean},{va_loss_mean}',
            os.path.join(output_dir, 'training_log.txt'), mode='a')
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
