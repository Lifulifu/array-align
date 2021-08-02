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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
                        output_dir=path, batch_no=batch, n_samples=5, eq='clahe')

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
                    output_dir=path, batch_no=batch, n_samples=5, eq='clahe')
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


def train_gridding_pipeline(datas, dataset, output_dir, model:str,
                            pretrained=True, start_epoch=1, epochs=2000, batch_size=64, save_interval=500, patience=0,
                            max_imgs=20, max_ori=0, augment=0, max_samples=0, lr=(0.001, 0.01), use_gal=True, device='cuda:0'):
    '''
        datas: list of csvs that record (img, gal, gpr) file paths
        max_imgs, max_ori, max_xs: max amount of imgs, real samples, total samples
    '''
    os.makedirs(output_dir, exist_ok=True)
    if model is None:
        if dataset.channel_mode == 'stack':
            model = Resnet(channels=2, pretrained=pretrained)
        else:
            model = Resnet(channels=1, pretrained=pretrained)
    else:
        model = torch.load(model)

    xtrs, xvas, ytrs, yvas = [], [], [], []
    for path in datas:
        files = read_csv(path)
        if not use_gal:
            for f in files:
                f[1] = ''
        if max_imgs > 0 and max_imgs < len(files):
            files = np.array(files)[np.random.choice(
                range(len(files)), max_imgs, replace=False)]

        xs, ys, isori = dataset.imgs2xy(
            files, augment=augment, keep_ori=True)
        xs, xaugs, ys, yaugs = xs[isori], xs[~isori], ys[isori], ys[~isori]
        print(xs.shape, ys.shape)

        xtr, xva, ytr, yva = data_split(
            xs, ys, .8, output_dir=output_dir, shuffle=False)
        xaugtr, xaugva, yaugtr, yaugva = data_split(
            xaugs, yaugs, .8, output_dir=output_dir, shuffle=False)
        if max_ori > 0 and len(xtr) > max_ori:
            idxs = np.random.choice(range(len(xtr)), size=max_ori, replace=False)
            xtr, ytr = xtr[idxs], ytr[idxs]
            aug_idxs = np.concatenate([np.arange(i, i+augment) for i in idxs])
            xaugtr, yaugtr = xaugtr[aug_idxs], yaugtr[aug_idxs]

        if max_samples > 0 and len(xtr) + len(xaugtr) > max_samples:
            aug_amount = max_samples - len(xtr)
            xaugtr, yaugtr = xaugtr[:aug_amount], yaugtr[:aug_amount]
        xtr, ytr = np.concatenate((xtr, xaugtr)), np.concatenate((ytr, yaugtr))
        xtrs.append(xtr)
        xvas.append(xva)
        ytrs.append(ytr)
        yvas.append(yva)

    xtrs = np.concatenate(xtrs)
    xvas = np.concatenate(xvas)
    ytrs = np.concatenate(ytrs)
    yvas = np.concatenate(yvas)
    print(xtrs.shape, ytrs.shape, xvas.shape, yvas.shape)

    train_block_corner_coord_model(model, xtrs, ytrs, xvas, yvas,
                                   epochs=epochs,
                                   start_epoch=start_epoch,
                                   lr=lr,
                                   batch_size=batch_size,
                                   output_dir=output_dir,
                                   device=device,
                                   save_interval=save_interval,
                                   patience=patience)


def pred_eval_gridding(data, output_dir, dataset, model, strict=True, write_img=False,
        find_match=False, finetune=False, device='cuda:0'):
    '''
    data (str): the csv path contains all (img, gal, gpr) data's paths
    '''

    os.makedirs(output_dir, exist_ok=True)
    files = read_csv(data)
    predictor = BlockCornerCoordPredictor(model, dataset, device=device)

    dists, hits = [], []
    for im, gal, gpr in files:
        print(im)
        gpr = Gpr(gpr)
        gal = Gal(gal) if gal != '' else make_fake_gal(gpr)

        df, spot_df = predictor.predict(im, gal, finetune=finetune)
        if df is None:
            print('prediction failed')
            continue

        # take results of the first channel
        df = df.groupby(['img_path', 'Block']).first()
        spot_df = spot_df.groupby(
            ['img_path', 'Block', 'Row', 'Column']).first()

        if strict:
            dist, hit = eval_gridding2(gpr, spot_df, find_match=find_match)
        else:
            dist, hit = eval_gridding(gpr, spot_df, find_match=find_match)
        dists.append(dist)
        hits.append(hit)

        if write_img is not False:
            im_name = im.replace('.tif', '').split('/')[-1]
            for channel, im in enumerate(read_tif(im, rgb=True, eq_method='clahe')):
                if write_img == 'block':
                    im = draw_corners_gpr(im, gal, gpr, color=(0, 255, 0))
                    im = draw_corners_df(im, df, color=(255, 0, 0))
                else:
                    im = draw_spots(
                        im, gpr.data[['X', 'Y']].values/dataset.pixel_size, color=(0, 255, 0))
                    im = draw_spots(
                        im, spot_df[['x', 'y']].values, color=(255, 0, 0))
                os.makedirs(output_dir, exist_ok=True)
                if not cv2.imwrite(
                        os.path.join(output_dir, f'{im_name}_{channel}.png'), im):
                    print('imwrite failed again OMFG')

    dist = np.concatenate(dists)
    hit = np.concatenate(hits)
    mae = dist.mean()
    acc = hit.sum() / len(hit)
    return acc, mae


def fewshot_train_reptile_pipeline(datasets, output_dir, model=None, **args):
    tasks = []
    tr_files = os.path.join(output_dir, 'tr.csv')
    va_files = os.path.join(output_dir, 'va.csv')
    if os.path.isfile(tr_files) and os.path.isfile(va_files):
        tr_tasks = read_csv(tr_files, table=False)
        va_tasks = read_csv(va_files, table=False)
    else:
        for ds in datasets:
            for task in os.listdir(datasets[ds]):
                tasks.append(os.path.join(datasets[ds], task))
        tr_tasks, va_tasks = train_test_split(
            tasks, test_size=.2, shuffle=True)
        write_file(tr_tasks, os.path.join(output_dir, 'tr.csv'))
        write_file(va_tasks, os.path.join(output_dir, 'va.csv'))
    reptile = Reptile(model, tr_tasks, va_tasks, output_dir=output_dir, **args)
    reptile.meta_train()


def fewshot_train_MAML_pipeline(datasets, output_dir, model, **args):
    tasks = []
    tr_files = os.path.join(output_dir, 'tr.csv')
    va_files = os.path.join(output_dir, 'va.csv')
    if os.path.isfile(tr_files) and os.path.isfile(va_files):
        tr_tasks = read_csv(tr_files, table=False)
        va_tasks = read_csv(va_files, table=False)
    else:
        for ds in datasets:
            for task in os.listdir(datasets[ds]):
                tasks.append(os.path.join(datasets[ds], task))
        tr_tasks, va_tasks = train_test_split(
            tasks, test_size=.2, shuffle=True)
        write_file(tr_tasks, os.path.join(output_dir, 'tr.csv'))
        write_file(va_tasks, os.path.join(output_dir, 'va.csv'))
    maml = MAML(model, tr_tasks, va_tasks, **args)
    maml.meta_train(output_dir)



