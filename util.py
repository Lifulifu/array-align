import numpy as np
import pandas as pd
import re
import cv2
from shapely.geometry import Polygon

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

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

def read_tif(path, rgb=False, eq_method=None, clahe_kwargs={'clipLimit': 20, 'tileGridSize': (8, 8)}):
    ims = cv2.imreadmulti(path)[1]
    if not ims:
        print(f'{path} failed to read.')
        return False
    for i, im in enumerate(ims):
        if eq_method:
            ims[i] = im_equalize(ims[i], method=eq_method, clahe_kwargs=clahe_kwargs)
        if rgb:
            ims[i] = cv2.cvtColor(ims[i], cv2.COLOR_GRAY2RGB)
    return ims

def im_equalize(im, method='clahe', clahe_kwargs={'clipLimit': 20, 'tileGridSize': (8, 8)}):
    if method == 'clahe':
        return cv2.createCLAHE(**clahe_kwargs).apply(im)
    return cv2.equalizeHist(im)

def im_morphology(im):
    kernel = np.ones((2,2), np.uint8)
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

def cat_im_batches_by_pad(im_list):
    maxlen = 0
    for im_batch in im_list:
        tmp = im_batch.shape[-2, -1].max()
        maxlen = tmp if tmp > maxlen else maxlen

def get_window_coords(block, gal, expand_rate=2., pixel_size=10):
    block_info = gal.header[f'Block{block}']
    cx, cy = block_info[Gal.CENTER_X], block_info[Gal.CENTER_Y]
    w = block_info[Gal.COL_MARGIN] * (block_info[Gal.N_COLS]-1) * expand_rate
    h = block_info[Gal.ROW_MARGIN] * (block_info[Gal.N_ROWS]-1) * expand_rate
    x1 = int((cx - w/2) / pixel_size)
    y1 = int((cy - h/2) / pixel_size)
    x2 = int((cx + w/2) / pixel_size)
    y2 = int((cy + h/2) / pixel_size)
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

def im_offset(img, x, y):
    mat = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))

def im_rotate(image, angle, center=None, scale=1.0):
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, mat, (w, h))

def get_ious(yb, ypredb):
    ious = []
    for y, ypred in zip(yb, ypredb):
        y, ypred = to_four_pts(y), to_four_pts(ypred)
        y, ypred = Polygon(y).convex_hull, Polygon(ypred).convex_hull
        inter_area = y.intersection(ypred).area
        iou = inter_area / (y.area + ypred.area - inter_area)
        ious.append(iou)
    return np.array(ious)

def is_iterable(x): # except str
    return (type(x) is list) or (type(x) is tuple)

def write_file(data, path, mode='w'):
    with open(path, mode) as f:
        if is_iterable(data):
            for row in data:
                if is_iterable(row):
                    f.write(','.join(row) + '\n')
                else:
                    f.write(row + '\n')
        else:
            f.write(data + '\n')

if __name__ == '__main__':
    img = read_tif('./data/kawasaki/fc/FC_3.tif')
    gpr = Gpr('./data/kawasaki/fc/FC_3.gpr')
    import IPython; IPython.embed(); exit()