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

from .util import *

class ImgGalGprDataset(Dataset):
    '''
    returns:
        x: img paths
        y: corresponding gpr paths
    '''
    def __init__(self, im_files, gal_paths, gpr_paths):
        assert len(im_files) == len(gal_paths) == len(gpr_paths)
        self.ims, self.gals, self.gprs = im_files, gal_paths, gpr_paths

    def __getitem__(self, i):
        return self.ims[i], self.gals[i], self.gprs[i]

    def __len__(self):
        return len(self.ims)

class AAADataset():
    def __init__(self, window_expand=2, down_sample=4, equalize=False, morphology=False, pixel_size=10):
        self.window_expand = window_expand
        self.down_sample = down_sample
        self.equalize = equalize
        self.morphology = morphology
        self.pixel_size = pixel_size

    def img2x(self, img_path, gal):
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

        xs, idxs = [], []
        for b in range(1, gal.header['BlockCount']+1):
            idxs.append((img_path, b))
            p1, p2 = get_window_coords(b, gal, expand_rate=self.window_expand)
            cropped = crop_window(im, p1, p2).astype('uint8')
            cropped = cv2.resize(cropped, (cropped.shape[1]//self.down_sample, cropped.shape[0]//self.down_sample))
            if self.equalize: cropped = im_equalize(cropped, method=self.equalize)
            if self.morphology: cropped = im_morphology(cropped)
            xs.append(cropped)

        xs = np.stack(xs)
        return np.expand_dims(xs, axis=1)/255, idxs

    def get_dataloaders(self, dirs, gal_paths, va_size=.2, te_size=.2, batch_size=1, save_dir=None):
        '''
        returns: (trainDataset, testDataset)
        '''
        self.gal_mem = dict() # read all gals for future reference
        for gal_path in gal_paths:
            if gal_path not in self.gal_mem:
                self.gal_mem[gal_path] = Gal(gal_path)

        ims, gals, gprs = [], [], []
        for d, g in zip(dirs, gal_paths):
            for f in os.listdir(d):
                if f.endswith('.tif'):
                    ims.append(os.path.join(d, f))
                    gprs.append(os.path.join(d, f.replace('.tif', '.gpr')))
                    gals.append(g)
        va_size = va_size / (1 - te_size) # split va from tr, so the correct proportion of va is va_size / tr_size
        imtr, imte, galtr, galte, gprtr, gprte = train_test_split(ims, gals, gprs, test_size=te_size, shuffle=True)
        imtr, imva, galtr, galva, gprtr, gprva = train_test_split(imtr, galtr, gprtr, test_size=va_size, shuffle=True)
        if save_dir:
            write_file(list(zip(imtr, galtr, gprtr)), os.path.join(save_dir, 'tr.txt'))
            write_file(list(zip(imva, galva, gprva)), os.path.join(save_dir, 'va.txt'))
            write_file(list(zip(imte, galte, gprte)), os.path.join(save_dir, 'te.txt'))
        return (
            DataLoader(ImgGalGprDataset(imtr, galtr, gprtr), batch_size=batch_size),
            DataLoader(ImgGalGprDataset(imva, galva, gprva), batch_size=batch_size),
            DataLoader(ImgGalGprDataset(imte, galte, gprte), batch_size=batch_size))

class BlockLCoordDataset(AAADataset):
    def __init__(self, window_expand=2, down_sample=4, equalize=False, morphology=False, pixel_size=10):
        super().__init__(window_expand, down_sample, equalize, morphology, pixel_size)

    def img2xy(self, img_path, gal, gpr_file, augment=True):
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

        ims = read_tif(img_path)
        if ims:
            im = np.stack(ims).max(axis=0)
        else:
            print(f'failed to read img {img_path}, skip.')
            return None, None, None

        gpr = Gpr(gpr_file) if type(gpr_file) == str else gpr_file
        xs, ys, idxs = [], [], []

        for b in range(1, gal.header['BlockCount']+1):
            idxs.append((img_path, b))
            n_rows = gal.header[f'Block{b}'][Gal.N_ROWS]
            n_cols = gal.header[f'Block{b}'][Gal.N_COLS]
            p1, p2 = get_window_coords(b, gal, expand_rate=self.window_expand)
            cropped = crop_window(im, p1, p2).astype('uint8')
            cropped = cv2.resize(cropped, (cropped.shape[1]//self.down_sample, cropped.shape[0]//self.down_sample))
            if self.equalize: cropped = im_equalize(cropped, method=self.equalize)
            if self.morphology: cropped = im_morphology(cropped)
            xs.append(cropped)

            y = np.array([
                gpr.data.loc[b, 1, 1]['X'] / self.pixel_size - p1[0],
                gpr.data.loc[b, 1, 1]['Y'] / self.pixel_size - p1[1],
                gpr.data.loc[b, n_rows, 1]['X'] / self.pixel_size - p1[0],
                gpr.data.loc[b, n_rows, 1]['Y'] / self.pixel_size - p1[1],
                gpr.data.loc[b, n_rows, n_cols]['X'] / self.pixel_size - p1[0],
                gpr.data.loc[b, n_rows, n_cols]['Y'] / self.pixel_size - p1[1]
            ]) / self.down_sample
            ys.append(y)

            if augment:
                fourth_pt = np.array([
                    gpr.data.loc[b, 1, n_cols]['X'] / self.pixel_size - p1[0],
                    gpr.data.loc[b, 1, n_cols]['Y'] / self.pixel_size - p1[1],
                ]) / self.down_sample
                flipx, flipy = self.flip_augment(cropped, np.concatenate((y, fourth_pt)))
                offsetx, offsety = self.offset_augment(cropped, y, n_samples=4)
                xs = xs + flipx + offsetx # list concat
                ys = ys + flipy + offsety

        xs = np.stack(xs)
        return np.expand_dims(xs, axis=1)/255, np.stack(ys), idxs

    def imgs2xy(self, img_paths, gal_paths, gpr_paths, augment=True):
        '''
        List of img files to xy for all blocks
        args: lists of img and gpr paths
        returns: aggregated x, y nparrays
        '''
        xs, ys, idxs = [], [], []
        for img_path, gal_path, gpr_path in zip(img_paths, gal_paths, gpr_paths):
            x, y, idx = self.img2xy(img_path, self.gal_mem[gal_path], gpr_path, augment=augment)
            if (x is not None) and (y is not None):
                xs.append(x)
                ys.append(y)
                idxs.append(idx)
        # pass if all cropped images are of same shape
        xs, ys = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
        idxs = np.concatenate(idxs, axis=0)
        return xs, ys, idxs

    def to_four_pts(self, pts):
        '''
        pts: 3 pts parellelgram
        returns: 4 pts (shape 4*2)
        '''
        if pts.shape != (3, 2): pts = pts.reshape(3, 2)
        fourth_pt = pts[2] + (pts[0] - pts[1])
        return np.concatenate((pts, fourth_pt.reshape(1, -1)), axis=0)

    def augment_im(self, im, pts, **kwargs):
        '''
        im: 2D np array
        '''
        flipxs, flipys = self.flip_augment(im, pts)
        augxs, augys = [], []
        for flipx, flipy in zip(flipxs, flipys):
            offsetxs, offsetys = self.offset_augment(flipx, flipy, **kwargs)
            augxs += offsetxs
            augys += offsetys
        augxs += flipxs
        augys += flipys
        return augxs, augys

    def offset_augment(self, im, pts, n_samples=1, max_offset=(.5, .5)):
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

    def flip_augment(self, im, pts):
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

class SpotCoordDataset(AAADataset):
    def __init__(self, window_expand=2, down_sample=4, equalize=False, morphology=False, pixel_size=10):
        super().__init__(window_expand, down_sample, equalize, morphology, pixel_size)

    def img2xy(self, img_path, gal, gpr_file, augment=True):
        '''
        Single img file to xys for all blocks and channels
        returns:
            xs:
                cropped window, grayscale[0~255]=>[0~1]
                shape ( #blocks, 1, win_shape[0], win_shape[1] )
            ys:
                imgs of same shape as xs
                1 = center of a spot of that block
                0 = background
            idx:
                (img_path, block) for one x sample
        '''
        ims = read_tif(img_path)
        if ims:
            im = np.stack(ims).max(axis=0)
        else:
            print(f'failed to read img {img_path}, skip.')
            return None, None, None

        gpr = Gpr(gpr_file) if type(gpr_file) == str else gpr_file
        xs, ys, idxs = [], [], []

        for b in range(1, gal.header['BlockCount']+1):
            idxs.append((img_path, b))
            n_rows = gal.header[f'Block{b}'][Gal.N_ROWS]
            n_cols = gal.header[f'Block{b}'][Gal.N_COLS]
            p1, p2 = get_window_coords(b, gal, expand_rate=self.window_expand)
            cropped = crop_window(im, p1, p2).astype('uint8')
            cropped = cv2.resize(cropped, (cropped.shape[1]//self.down_sample, cropped.shape[0]//self.down_sample))
            if self.equalize: cropped = im_equalize(cropped, method=self.equalize)
            if self.morphology: cropped = im_morphology(cropped)
            xs.append(cropped)

            y = np.zeros(cropped.shape)
            for idx, df_row in gpr.data.loc[b].iterrows():
                x_idx = int((df_row['X'] / self.pixel_size - p1[0]) / self.down_sample)
                y_idx = int((df_row['Y'] / self.pixel_size - p1[1]) / self.down_sample)
                y[y_idx, x_idx] = 1
            ys.append(y)

            # if augment:
            #     fourth_pt = np.array([
            #         gpr.data.loc[b, 1, n_cols]['X'] / self.pixel_size - p1[0],
            #         gpr.data.loc[b, 1, n_cols]['Y'] / self.pixel_size - p1[1],
            #     ]) / self.down_sample
            #     flipx, flipy = self.flip_augment(cropped, np.concatenate((y, fourth_pt)))
            #     offsetx, offsety = self.offset_augment(cropped, y, n_samples=4)
            #     xs = xs + flipx + offsetx # list concat
            #     ys = ys + flipy + offsety

        xs = np.stack(xs)
        return np.expand_dims(xs, axis=1)/255, np.stack(ys), idxs

    def imgs2xy(self, img_paths, gal_paths, gpr_paths, augment=True):
        '''
        List of img files to xy for all blocks
        args: lists of img and gpr paths
        returns: aggregated x, y nparrays
        '''
        xs, ys, idxs = [], [], []
        for img_path, gal_path, gpr_path in zip(img_paths, gal_paths, gpr_paths):
            x, y, idx = self.img2xy(img_path, self.gal_mem[gal_path], gpr_path, augment=augment)
            if (x is not None) and (y is not None):
                xs.append(x)
                ys.append(y)
                idxs.append(idx)
        # pass if all cropped images are of same shape
        xs, ys = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
        idxs = np.concatenate(idxs, axis=0)
        return xs, ys, idxs

if '__main__' == __name__:
    pass

