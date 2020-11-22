import cv2
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pprint import pprint

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from .util import write_file

class Gal():
    # index info of gal.header['Blockn']
    CENTER_X = 0; CENTER_Y = 1; SIZE = 2
    N_COLS = 3; COL_MARGIN = 4; N_ROWS = 5; ROW_MARGIN = 6
    def __init__(self, path):
        with open(path, 'rb') as f:
            data = f.readlines()
            data = [row.decode('utf-8', 'ignore').replace('"', '').strip() for row in data]
        # data part
        data_start_i = [i for i in range(len(data)) if data[i].startswith('Block\tColumn\tRow')][0]
        self.data = [ row.split('\t') for row in data[data_start_i:] ]
        self.data = pd.DataFrame(self.data[1:], columns=self.data[0])
        self.data = self.data.astype({
            'Block': int, 'Column': int, 'Row': int})
        self.data = self.data.set_index(['Block', 'Row', 'Column'])

        # header part
        self.header = {}
        for row in data[:data_start_i]:
            key_val = row.split('=')
            if len(key_val) == 2:
                k, v = key_val
                self.header[k] = v.strip()
                try:
                    self.header[k] = int(self.header[k])
                except:
                    pass
                if re.search('Block\d+', k) is not None:
                    self.header[k] = [int(x.strip()) for x in v.split(',')]

class Gpr():
    def __init__(self, path):
        with open(path, 'rb') as f:
            data = f.readlines()
            data = [row.decode('utf-8', 'ignore').replace('"', '').strip() for row in data]
        # data part
        data_start_i = [i for i in range(len(data)) if data[i].startswith('Block\tColumn\tRow')][0]
        self.data = [ row.split('\t') for row in data[data_start_i:] ]
        self.data = pd.DataFrame(self.data[1:], columns=self.data[0])
        self.data = self.data.astype({
            'Block': int, 'Column': int, 'Row': int, 'X': int, 'Y': int})
        self.data = self.data.set_index(['Block', 'Row', 'Column'])

        # header part
        self.header = {}
        for row in data[:data_start_i]:
            key_val = row.split('=')
            if len(key_val) == 2:
                k, v = key_val
                self.header[k] = v.strip()
                try:
                    self.header[k] = int(self.header[k])
                except:
                    pass

class ArrayAlignDataset(Dataset):
    '''
    returns:
        x: img paths
        y: corresponding gpr paths
    '''
    def __init__(self, im_files, gpr_files, down_sample=2):
        assert len(im_files) == len(gpr_files)
        self.ims, self.gprs = im_files, gpr_files

    def __getitem__(self, i):
        return self.ims[i], self.gprs[i]

    def __len__(self):
        return len(self.ims)

def read_tif(path, rgb=False, eq_method='clahe', clahe_kwargs={'clipLimit': 20, 'tileGridSize': (8, 8)}):
    ims = cv2.imreadmulti(path)[1]
    if not ims:
        print(f'{path} failed to read.')
        return False
    for i, im in enumerate(ims):
        if eq_method:
            ims[i] = im_equalize(ims[i], method=eq_method, clahe_kwargs=clahe_kwargs)
        if rgb:
            ims[i] = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    return ims

def im_equalize(im, method='clahe', clahe_kwargs={'clipLimit': 20, 'tileGridSize': (8, 8)}):
    if method == 'clahe':
        return cv2.createCLAHE(**clahe_kwargs).apply(im)
    else: # simple equalization
        return cv2.equalizeHist(im)

def im_morphology(im):
    kernel = np.ones((2,2), np.uint8)
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

def get_window_coords(block, gal, expand_rate=2.):
    block_info = gal.header[f'Block{block}']
    cx, cy = block_info[Gal.CENTER_X], block_info[Gal.CENTER_Y]
    w = block_info[Gal.COL_MARGIN] * (block_info[Gal.N_COLS]-1) * expand_rate
    h = block_info[Gal.ROW_MARGIN] * (block_info[Gal.N_ROWS]-1) * expand_rate
    x1 = int((cx - w/2)*.1)
    y1 = int((cy - h/2)*.1)
    x2 = int((cx + w/2)*.1)
    y2 = int((cy + h/2)*.1)
    return (x1, y1), (x2, y2)

def crop_window(im, p1, p2, pad_val=0):
    # it is assumed that if p1 is out of bounds, then p2 will not, vice versa
    winh, winw = p2[1]-p1[1], p2[0]-p1[0]
    h, w = im.shape
    xstart, ystart, xend, yend = 0, 0, w, h
    if p1[0] < 0:
        xstart = -p1[0]
    if p1[1] < 0:
        ystart = -p1[1]
    if p2[0] >= w:
        xend = w - p1[0]
    if p2[1] >= h:
        yend = h - p1[1]
    cropped = im[
        np.clip(p1[1], 0, h) : np.clip(p2[1], 0, h),
        np.clip(p1[0], 0, w) : np.clip(p2[0], 0, w)
    ]
    padded = np.full((winh, winw), pad_val)
    padded[ystart:yend, xstart:xend] = cropped
    return padded

def augment_im(im, pts, **kwargs):
    flipxs, flipys = flip_augment(im, pts)
    augxs, augys = [], []
    for flipx, flipy in zip(flipxs, flipys):
        offsetxs, offsetys = offset_augment(flipx, flipy, **kwargs)
        augxs += offsetxs
        augys += offsetys
    augxs += flipxs
    augys += flipys
    return augxs, augys

def offset_augment(im, pts, n_samples=1, max_offset=(.5, .5)):
    '''
    pts: 3 points
    max_offset: (x, y)
    '''
    pts = pts.reshape(3, 2)
    augx, augy = [], []
    h, w = im.shape
    max_offset = np.array(max_offset) * np.array([w, h])
    for i in range(100):
        rand = np.random.rand(2) * 2 - 1
        offset = rand * max_offset
        offset_pts = (pts + offset).astype(int)
        if (offset_pts.min()) >= 0 and (offset_pts[:, 0].max() < w) and (offset_pts[:, 1].max() < h):
            m = np.array([[1, 0, offset[0]],
                          [0, 1, offset[1]]])
            augx.append(cv2.warpAffine(im, m, (w, h))) # offset transform
            augy.append(offset_pts.flatten())
            if len(augx) >= n_samples:
                return augx, augy
    return augx, augy

def flip_augment(im, pts):
    '''
    pts: 4 points: (x1, y1, ... x4, y4)
    '''
    pts = pts.reshape(4, 2)
    augx, augy = [], []
    h, w = im.shape
    # horizontal flip
    augx.append(cv2.flip(im, 1))
    augy.append(np.array([
        w - pts[3, 0],
        pts[3, 1],
        w - pts[2, 0],
        pts[2, 1],
        w - pts[1, 0],
        pts[1, 1]
    ]))
    # vertical flip
    augx.append(cv2.flip(im, 0))
    augy.append(np.array([
        pts[1, 0],
        h - pts[1, 1],
        pts[0, 0],
        h - pts[0, 1],
        pts[3, 0],
        h - pts[3, 1]
    ]))
    # vertical + horizontal
    augx.append(cv2.flip(im, -1))
    augy.append(np.array([
        w - pts[2, 0],
        h - pts[2, 1],
        w - pts[3, 0],
        h - pts[3, 1],
        w - pts[0, 0],
        h - pts[0, 1]
    ]))
    return augx, augy # list of arrays

def img2x(img_path, gal_file, window_expand=2, down_sample=4, equalize='clahe', morphology=True):
    '''
    Single img file to xs for all blocks and fused channels

    returns:
        xs:
            cropped window, grayscale[0~255]=>[0~1]
            shape ( #blocks, 1, win_shape[0], win_shape[1] )
        idx:
            (img_path, block) for one x sample
    '''
    ims = read_tif(img_path)
    if ims:
        im = np.stack(ims).max(axis=0) # fuse img channels
    else:
        print(f'failed to read img {img_path}, skip.')
        return None, None

    gal = Gal(gal_file) if type(gal_file) == str else gal_file
    xs, idxs = [], []
    for b in range(1, gal.header['BlockCount']+1):
        idxs.append((img_path, b))

        p1, p2 = get_window_coords(b, gal, expand_rate=window_expand)
        cropped = crop_window(im, p1, p2).astype('uint8')
        cropped = cv2.resize(cropped, (cropped.shape[1]//down_sample, cropped.shape[0]//down_sample))
        if equalize: cropped = im_equalize(cropped, method=equalize)
        if morphology: cropped = im_morphology(cropped)
        xs.append(cropped)

    xs = np.stack(xs)
    return np.expand_dims(xs, axis=1)/255, idxs

def img2xy(im_file, gal_file, gpr_file, window_expand=2, down_sample=4,
        augment=True, equalize='clahe', morphology=True):
    '''
    Single img file to xys for all blocks and channels

    returns:
        xs:
            cropped window, grayscale[0~255]=>[0~1]
            shape ( #blocks, 1, win_shape[0], win_shape[1] )
        ys:
            top left, bottom left, bottom right XYs of a block (L shape)
            (relative to window coords)
            shape ( #blocks*2, 6 )
        idx:
            (img_path, block) for one x sample
    '''

    ims = read_tif(im_file)
    if ims:
        im = np.stack(ims).max(axis=0)
    else:
        print(f'failed to read img {im_file}, skip.')
        return None, None

    gal = Gal(gal_file) if type(gal_file) == str else gal_file
    gpr = Gpr(gpr_file) if type(gpr_file) == str else gpr_file
    xs, ys, idxs = [], [], []

    for b in range(1, gal.header['BlockCount']+1):

        idxs.append((im_file, b))
        n_rows = gal.header[f'Block{b}'][Gal.N_ROWS]
        n_cols = gal.header[f'Block{b}'][Gal.N_COLS]
        p1, p2 = get_window_coords(b, gal, expand_rate=window_expand)
        cropped = crop_window(im, p1, p2).astype('uint8')
        cropped = cv2.resize(cropped, (cropped.shape[1]//down_sample, cropped.shape[0]//down_sample))
        if equalize: cropped = im_equalize(cropped, method=equalize)
        if morphology: cropped = im_morphology(cropped)
        xs.append(cropped)

        y = np.array([
            gpr.data.loc[b, 1, 1]['X']*.1 - p1[0],
            gpr.data.loc[b, 1, 1]['Y']*.1 - p1[1],
            gpr.data.loc[b, n_rows, 1]['X']*.1 - p1[0],
            gpr.data.loc[b, n_rows, 1]['Y']*.1 - p1[1],
            gpr.data.loc[b, n_rows, n_cols]['X']*.1 - p1[0],
            gpr.data.loc[b, n_rows, n_cols]['Y']*.1 - p1[1]
        ]) / down_sample
        ys.append(y)

        if augment:
            fourth_pt = np.array([
                gpr.data.loc[b, 1, n_cols]['X']*.1 - p1[0],
                gpr.data.loc[b, 1, n_cols]['Y']*.1 - p1[1],
            ]) / down_sample
            flipx, flipy = flip_augment(cropped, np.concatenate((y, fourth_pt)))
            offsetx, offsety = offset_augment(cropped, y, n_samples=4)
            xs = xs + flipx + offsetx # list concat
            ys = ys + flipy + offsety

    xs = np.stack(xs)
    return np.expand_dims(xs, axis=1)/255, np.stack(ys), idxs

def imgs2xy(ims, gal_file, gpr_files, **kwargs):
    '''
    List of img files to xy for all blocks

    args: lists of img and gpr paths
    returns: aggregated x, y nparrays
    '''
    xs, ys = [], []
    gal = Gal(gal_file)
    for im, gpr_file in zip(ims, gpr_files):
        x, y, _ = img2xy(im, gal, gpr_file, **kwargs)
        if x is not None:
            xs.append(x)
            ys.append(y)
        else:
            return None, None
    xs, ys = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
    return xs, ys

def get_datasets(dirs, va_size=.2, te_size=.2, save_dir=None):
    '''
    returns: (trainDataset, testDataset)
    '''
    ims, gprs = [], []
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith('.tif'):
                ims.append(d + f)
                gprs.append(d + f.replace('.tif', '.gpr'))
    va_size = va_size / (1 - te_size) # split va from tr, so the correct proportion of va is va_size / tr_size
    imtr, imte, gprtr, gprte = train_test_split(ims, gprs, test_size=te_size, shuffle=True)
    imtr, imva, gprtr, gprva = train_test_split(imtr, gprtr, test_size=va_size, shuffle=True)
    if save_dir:
        write_file(imtr, os.path.join(save_dir, 'tr.txt'))
        write_file(imva, os.path.join(save_dir, 'va.txt'))
        write_file(imte, os.path.join(save_dir, 'te.txt'))
    return ArrayAlignDataset(imtr, gprtr), ArrayAlignDataset(imva, gprva), ArrayAlignDataset(imte, gprte)



if '__main__' == __name__:
    pass

