import torch
import torch.nn as nn
import cv2
import os
from tqdm import tqdm
import json

from .model import Resnet
from .preprocess import *
from .draw import *
from .util import *

def train(trdl, vadl, gal_file, down_sample=4, window_expand=2, eq=None,
    morphology=None, model=None, epochs=100, start_epoch=0, minibatch=None,
    save_interval=5, output_dir='outputs/', device='cuda:0'):

    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'preds'), exist_ok=True)
    if model == None:
        model = Resnet().to(device)
    else:
        model = model.to(device)

    loss_func = nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # log file header
    write_file('epoch,tr_loss,va_loss',
            os.path.join(output_dir, 'training_log.txt'), mode='w')

    for epoch in range(start_epoch, start_epoch+epochs):
        print(f'\nepoch {epoch}')

        trdl_bar = tqdm(trdl, ncols=100)
        for ims, gprs in trdl_bar:
            xb, yb = imgs2xy(ims, gal_file, gprs, augment=True, equalize=eq,
                down_sample=down_sample, window_expand=window_expand)
            # draw_xy(xb, yb); exit()
            if xb is None:
                print("empty batch"); continue
            if minibatch:
                idx = np.random.randint(low=0, high=len(xb), size=minibatch)
                xb, yb = xb[idx], yb[idx]
            xb, yb = torch.tensor(xb).float().to(device), torch.tensor(yb).float().to(device)
            ypredb = model(xb)
            loss = loss_func(ypredb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss = loss.item()
            trdl_bar.set_description(f'tr loss: {tr_loss:.3f}')

        vadl_bar = tqdm(vadl, ncols=100)
        with torch.no_grad():
            for ims, gprs in vadl_bar:
                xb, yb = imgs2xy(ims, gal_file, gprs, augment=True, equalize=eq,
                    down_sample=down_sample, window_expand=window_expand, morphology=morphology)
                if xb is None:
                    print("empty batch")
                    continue
                if minibatch:
                    idx = np.random.randint(low=0, high=len(xb), size=minibatch)
                    xb, yb = xb[idx], yb[idx]
                xb, yb = torch.tensor(xb).float().to(device), torch.tensor(yb).float().to(device)
                ypredb = model(xb)
                loss = loss_func(ypredb, yb)
                va_loss = loss.item()
                vadl_bar.set_description(f'va loss: {va_loss:.3f}')

            if epoch % save_interval == 0 and xb is not None:
                torch.save(model, os.path.join(output_dir, f'models/{epoch}.pt'))
                sample_imgs(xb, yb, ypredb,
                    output_dir=os.path.join(output_dir, f'preds/e{epoch}/'))

        # epoch end
        write_file(f'{epoch},{tr_loss},{va_loss}',
            os.path.join(output_dir, 'training_log.txt'), mode='a')

def sample_imgs(xb, yb, ypredb, output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, (x, y, ypred) in enumerate(zip(xb, yb, ypredb)):
            im = np.expand_dims(x.cpu().numpy()[0], -1) * 255
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            y = np.round(y.cpu().numpy()).astype(int)
            ypred = np.round(ypred.cpu().numpy()).astype(int)
            im = draw_parallelogram(im, y, color=(0,255,0))
            im = draw_parallelogram(im, ypred, color=(255,0,0))
            cv2.imwrite(os.path.join(output_dir, f'{i}.png'), im)

def test_draw():
    for i in range(1, 11):
        gal = Gal('data/20200820 KD focus (12)/kd.GAL')
        gpr = Gpr(f'data/20200820 KD focus (12)/{i}.gpr')
        im = read_tif(f'data/20200820 KD focus (12)/{i}.tif', rgb=True)[0]
        im = draw_gt_blocks(im, gal, gpr, color=(0,255,0))
        im = draw_gal_centers(im, gal, color=(0,255,255))
        im = draw_windows(im, gal, expand_rate=2, color=(0,255,255))
        cv2.imwrite(f'test/gt_{i}.png', im)
