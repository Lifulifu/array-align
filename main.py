import torch
import torch.nn as nn
import cv2
import os

from model import Resnet
from preprocess import *
from draw import *
# import IPython; IPython.embed(); exit()

def train(data_dirs, gal_file, model=None, epochs=100, start_epoch=0, save_interval=5,
          model_dir='models/', result_dir='pred/'):

    os.makedirs(model_dir, exist_ok=True)
    if model == None:
        model = Resnet().cuda()
    trds, teds = get_datasets(data_dirs, test_size=.1)
    trdl = DataLoader(trds, batch_size=1, shuffle=True)
    tedl = DataLoader(teds, batch_size=1, shuffle=True)
    print('train:', len(trds), 'test', len(teds))

    loss_func = nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(start_epoch, start_epoch+epochs):
        print(f'\nepoch {epoch}')
        for batch, (ims, gprs) in enumerate(trdl):
            xb, yb = load_batch(ims, gal_file, gprs, augment=True, down_sample=4)
            # draw_xy(xb, yb); exit()
            xb, yb = xb.cuda(), yb.cuda()
            ypredb = model(xb)
            loss = loss_func(ypredb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'tr loss: {loss.item()}')

        for batch, (ims, gprs) in enumerate(tedl):
            xb, yb = load_batch(ims, gal_file, gprs, augment=True, down_sample=4)
            xb, yb = xb.cuda(), yb.cuda()
            ypredb = model(xb)
            loss = loss_func(ypredb, yb)
        print(f'te loss: {loss.item()}')

        if epoch % save_interval == 0:
            sample_imgs(xb[:10], yb[:10], ypredb[:10],
                output_dir=os.path.join(result_dir, f'e{epoch}b{batch}/'))
            torch.save(model, os.path.join(model_dir, f'{epoch}.pt'))

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

if '__main__' == __name__:
    # m = torch.load('models/95.pt').cuda()
    train(['data/20200820 KD focus (12)/'],
          gal_file='data/20200820 KD focus (12)/kd.GAL',
          epochs=200, start_epoch=0, save_interval=10,
          model_dir='models/kd_focus_res18_aug/',
          result_dir='pred/kd_focus_res18_aug/')
    # test_draw()