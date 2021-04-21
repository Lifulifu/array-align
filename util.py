import numpy as np
import pandas as pd
import re
import json
import cv2
from shapely.geometry import Polygon
import os

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class Gal():
    # index info of gal.header['Blockn']
    CENTER_X = 0
    CENTER_Y = 1
    SIZE = 2
    N_COLS = 3
    COL_MARGIN = 4
    N_ROWS = 5
    ROW_MARGIN = 6

    def __init__(self, path=None, fake=False):
        if not fake:
            self.read_gal(path)
        else:
            self.data = []
            self.header = dict()

    def read_gal(self, path):
        with open(path, 'rb') as f:
            data = f.readlines()
            data = [row.decode('utf-8', 'ignore').replace('"',
                                                          '').strip() for row in data]
        # data part
        data_start_i = [i for i in range(
            len(data)) if data[i].startswith('Block\tColumn\tRow')][0]
        self.data = [row.split('\t') for row in data[data_start_i:]]
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
            data = [row.decode('utf-8', 'ignore').replace('"',
                                                          '').strip() for row in data]
        # data part
        data_start_i = [i for i in range(
            len(data)) if data[i].startswith('Block\tColumn\tRow')][0]
        self.data = [row.split('\t') for row in data[data_start_i:]]
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


def read_tif(path, rgb=False, eq_method=None, channel_mode=None,
        clahe_kwargs={'clipLimit': 20, 'tileGridSize': (8, 8)}):
    ims = cv2.imreadmulti(path)[1]
    if not ims:
        print(f'{path} failed to read.')
        return False
    for i, im in enumerate(ims):
        if eq_method:
            ims[i] = im_equalize(ims[i], method=eq_method,
                                 clahe_kwargs=clahe_kwargs)
        if rgb:
            ims[i] = cv2.cvtColor(ims[i], cv2.COLOR_GRAY2RGB)
    if channel_mode == 'max':
        return np.clip(np.stack(ims).max(axis=0), 0, 255)
    elif channel_mode == 'stack':
        return np.stack(ims)

    return ims


def make_fake_gal(gpr):
    '''
    gpr: type Gpr
    '''
    result = Gal(fake=True)
    result.header['BlockCount'] = gpr.data.index.get_level_values(
        'Block').max()
    for b in range(1, result.header['BlockCount']+1):
        info = [0] * 7  # one block header contains 7 numbers
        nrows = gpr.data.loc[b].index.get_level_values('Row').max()
        ncols = gpr.data.loc[b].index.get_level_values('Column').max()
        start = gpr.data.loc[b, 1, 1][['X', 'Y']].values
        end = gpr.data.loc[b, nrows, ncols][['X', 'Y']].values
        rand_center = start + (end - start) * np.random.uniform(.2, .8)

        dxy = gpr.data.loc[b, nrows, 1][['X', 'Y']].values - \
            gpr.data.loc[b, 1, 1][['X', 'Y']].values
        row_margin = np.sqrt(np.sum(dxy ** 2)) // nrows
        dxy = gpr.data.loc[b, 1, ncols][['X', 'Y']].values - \
            gpr.data.loc[b, 1, 1][['X', 'Y']].values
        col_margin = np.sqrt(np.sum(dxy ** 2)) // ncols

        info[Gal.N_ROWS] = nrows
        info[Gal.N_COLS] = ncols
        info[Gal.ROW_MARGIN] = row_margin
        info[Gal.COL_MARGIN] = col_margin
        info[Gal.CENTER_X] = rand_center[0]
        info[Gal.CENTER_Y] = rand_center[1]
        result.header[f'Block{b}'] = info
    return result


def im_equalize(im, method='clahe', clahe_kwargs={'clipLimit': 20, 'tileGridSize': (8, 8)}):
    im = im.astype(np.uint8)
    if method == 'clahe':
        return cv2.createCLAHE(**clahe_kwargs).apply(im)
    return cv2.equalizeHist(im)


def im_morphology(im):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)


def im_contain_resize(im, size):
    '''
    input shape (h, w), output shape (size)
    '''
    if im.shape[0] > im.shape[1]:
        scale = min(size) / im.shape[0]
        im = cv2.resize(
            im, (int(im.shape[1] * scale), min(size)))
    else:
        scale = min(size) / im.shape[1]
        im = cv2.resize(
            im, (min(size), int(im.shape[0] * scale)))
    result = np.zeros((size[1], size[0]))
    result[:im.shape[0], :im.shape[1]] = im
    return result, scale


def get_window_coords(gal, b, expand=None, size=None, pixel_size=10):
    block_info = gal.header[f'Block{b}']
    cx, cy = block_info[Gal.CENTER_X], block_info[Gal.CENTER_Y]
    if expand:
        w = block_info[Gal.COL_MARGIN] * (block_info[Gal.N_COLS]-1) * expand
        h = block_info[Gal.ROW_MARGIN] * (block_info[Gal.N_ROWS]-1) * expand
    else:
        w, h = size
    x1 = int((cx - w/2) / pixel_size)
    y1 = int((cy - h/2) / pixel_size)
    x2 = int((cx + w/2) / pixel_size)
    y2 = int((cy + h/2) / pixel_size)
    return (x1, y1), (x2, y2)


def crop_window(im, p1, p2, pad_val=0):
    # it is assumed that if p1 is out of bounds, then p2 will not, vice versa
    winh, winw = p2[1]-p1[1], p2[0]-p1[0]
    h, w = im.shape[-2:]
    xstart, ystart, xend, yend = 0, 0, w, h
    if p1[0] < 0:
        xstart = -p1[0]
    if p1[1] < 0:
        ystart = -p1[1]
    if p2[0] >= w:
        xend = w - p1[0]
    if p2[1] >= h:
        yend = h - p1[1]

    if len(im.shape) > 2:  # shape (c, h, w)
        cropped = im[:,
            np.clip(p1[1], 0, h): np.clip(p2[1], 0, h),
            np.clip(p1[0], 0, w): np.clip(p2[0], 0, w)]
        padded = np.full((im.shape[0], winh, winw), pad_val)
        padded[:, ystart:yend, xstart:xend] = cropped
    else:
        cropped = im[
            np.clip(p1[1], 0, h): np.clip(p2[1], 0, h),
            np.clip(p1[0], 0, w): np.clip(p2[0], 0, w)]
        padded = np.full((winh, winw), pad_val)
        padded[ystart:yend, xstart:xend] = cropped

    return padded


def im_offset(img, x, y):
    mat = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))


def im_rotate(image, angle, center=None, scale=1.0):
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, mat, (w, h))


def is_listlike(x):  # except str
    return (type(x) is list) or (type(x) is tuple)


def write_file(data, path, mode='w'):
    with open(path, mode) as f:
        if is_listlike(data):
            for row in data:
                if is_listlike(row):
                    # might contain gal path = None
                    row = [str(item) for item in row]
                    f.write(','.join(row) + '\n')
                else:
                    f.write(str(row) + '\n')
        else:
            f.write(str(data) + '\n')


def write_dict(d, path):
    with open(path, 'w') as f:
        f.write(json.dumps(d))


def eval_gridding(gpr, pred, gal=None, mode='spot', tolerance=.5,
        gpr_coords=['X', 'Y'], pred_coords=['x', 'y'], pixel_size=10):
    '''
    assume coord information of 1 image (chip)

    pred_df: block corner coord or spot coord
    mode: 'spot' or 'block'
    tolerance: for acc calculation
               False spot if distance > tolerance * min(row margin, col margin)
    '''
    if gal is None:
        gal = make_fake_gal(gpr)
    # rename columns to avoid auto rename while merging
    for i, (gpr_coord, pred_coord) in enumerate(zip(gpr_coords, pred_coords)):
        if gpr_coord == pred_coord:
            pred = pred.rename(columns={pred_coord: pred_coord+'_'})
            pred_coords[i] = pred_coord+'_'

    col_thres = tolerance * gal.header['Block1'][Gal.COL_MARGIN] // pixel_size
    row_thres = tolerance * gal.header['Block1'][Gal.ROW_MARGIN] // pixel_size
    df = pd.merge(gpr.data, pred, on=['Block', 'Row', 'Column'], how='left')
    gt, pred = df[gpr_coords].values // pixel_size, df[pred_coords].values
    dist = np.abs(gt - pred)
    hits = (dist[:, 0] <= col_thres) & (dist[:, 1] <= row_thres)
    return dist, hits


def data_split(x, y, tr_size, va_size=None, shuffle=True, output_dir=None):
    '''
    tr te split and save idxs to output_dir
    '''
    idxs = list(range(len(x)))
    if shuffle:
        np.random.shuffle(idxs)
    tr_size = int(len(x) * tr_size)
    tr_idx = idxs[:tr_size]
    if va_size is not None:
        va_size = int(len(x) * va_size)
        va_idx = idxs[tr_size : tr_size+va_size]
        te_idx = idxs[tr_size+va_size:]
    else:
        va_idx = idxs[tr_size:]
    if output_dir:
        write_file(tr_idx, os.path.join(output_dir, 'tr.csv'))
        write_file(va_idx, os.path.join(output_dir, 'va.csv'))
        if va_size is not None:
            write_file(te_idx, os.path.join(output_dir, 'te.csv'))
    if va_size is not None:
        return x[tr_idx], x[va_idx], x[te_idx], y[tr_idx], y[va_idx], y[te_idx]
    return x[tr_idx], x[va_idx], y[tr_idx], y[va_idx]


def read_csv(csv, table=True):
    '''
    table: if True, read as 2D list, else 1D
    '''
    with open(csv) as f:
        data = f.readlines()
        if table:
            data = [line.strip().split(',') for line in data]
        else:
            data = [line.strip() for line in data]
    return data


def read_dir_files(dirs, gal_path:str, exclude=[]):
    results = []
    for d in dirs:
        for f in os.listdir(d):
            f = os.path.join(d, f)
            if f.endswith('.tif') and not (f in exclude):
                results.append((f, gal_path, f.replace('.tif', '.gpr')))
    return results


if __name__ == '__main__':
    import IPython
    IPython.embed()
    exit()
