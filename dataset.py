import cv2
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pprint import pprint
import imgaug
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as aug

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

class ArrayBlockDataset():
    def __init__(self, window_expand=2, down_sample=4, equalize=False,
            clahe_kwargs={'clipLimit': 20, 'tileGridSize': (8, 8)}, morphology=False, pixel_size=10):
        self.window_expand = window_expand
        self.down_sample = down_sample
        self.equalize = equalize
        self.clahe_kwargs = clahe_kwargs
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
            if self.equalize: cropped = im_equalize(cropped, method=self.equalize, clahe_kwargs=self.clahe_kwargs)
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

class BlockCornerCoordDataset(ArrayBlockDataset):
    def __init__(self, window_expand=2, down_sample=4, equalize=False, morphology=False, pixel_size=10):
        super().__init__(window_expand, down_sample, equalize, morphology, pixel_size)
        self.aug_seq = aug_seq = aug.Sequential([
            aug.HorizontalFlip(0.5),
            aug.VerticalFlip(0.5),
            aug.Sometimes(0.5, aug.GaussianBlur(sigma=(0, 0.2))),
            aug.LinearContrast((0.8, 1.2)),
            aug.AdditiveGaussianNoise(scale=(0.0, 0.05*255)),
            aug.Multiply((0.8, 1.2)),
            aug.Affine(
                scale=(0.7, 1),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-10, 10),
            )
        ], random_order=True)

    def img2xy(self, img_path, gal, gpr, augment=0):
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
        imgs, idxs = self.img2x(img_path, gal) # imgs: (N, 1, h, w)
        h, w = imgs.shape[2], imgs.shape[3]
        gpr = Gpr(gpr) if type(gpr) == str else gpr
        xs, ys = [], []
        for img, (img_path, b) in zip(imgs, idxs):
            nrows = gal.header[f'Block{b}'][Gal.N_ROWS]
            ncols = gal.header[f'Block{b}'][Gal.N_COLS]
            p1, p2 = get_window_coords(b, gal, expand_rate=self.window_expand)
            # minus window offset
            block_df = (gpr.data.loc[b][['X', 'Y']] / self.pixel_size - np.array(p1)) / self.down_sample
            kpts = KeypointsOnImage([
                Keypoint(x=block_df.loc[1, 1]['X'], y=block_df.loc[1, 1]['Y']),
                Keypoint(x=block_df.loc[1, ncols]['X'], y=block_df.loc[1, ncols]['Y']),
                Keypoint(x=block_df.loc[nrows, ncols]['X'], y=block_df.loc[nrows, ncols]['Y']),
                Keypoint(x=block_df.loc[nrows, 1]['X'], y=block_df.loc[nrows, 1]['Y'])
            ], shape=(h, w))

            if augment <= 0:
                xs.append(img)
                coord = self.to_Lcoord(kpts.to_xy_array())
                ys.append(coord.flatten())
            else:
                for i in range(augment):
                    img_aug, kpts_aug = self.aug_seq(image=(img[0]*255).astype('uint8'), keypoints=kpts) # img: (1, w, h) -> (w, h)
                    coord = self.to_Lcoord(kpts_aug.to_xy_array()) # (3, 2)
                    if (coord[:, 0].max() >= w) or (coord[:, 1].max() >= h) or (coord[:, 0].min() < 0) or (coord[:, 1].min() < 0):
                        continue # skip if coord out of bounds
                    xs.append(np.array([img_aug/255]))
                    ys.append(coord.flatten())

        return np.stack(xs), np.stack(ys)

    def imgs2xy(self, img_paths, gal_paths, gpr_paths, augment=True):
        '''
        List of img files to xy for all blocks
        args: lists of img and gpr paths
        returns: aggregated x, y nparrays
        '''
        xs, ys = [], []
        for img_path, gal_path, gpr_path in zip(img_paths, gal_paths, gpr_paths):
            x, y = self.img2xy(img_path, self.gal_mem[gal_path], gpr_path, augment=augment)
            if (x is not None) and (y is not None):
                xs.append(x)
                ys.append(y)
        # pass if all cropped images are of same shape
        xs, ys = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
        return xs, ys

    def to_Lcoord(self, coord):
        '''
        coord: (4, 2) not sorted -> (3, 2) sorted L coord
        '''
        coord = coord[np.argsort(coord[:, 0])] # sort by x
        coord[:2] = coord[:2][np.argsort(coord[:2][:, 1])] # left 2 points sort by y
        coord[2:] = coord[2:][np.argsort(coord[2:][:, 1])] # right 2 points sort by y
        return coord[[0, 1, 3]] # top left, bottom left, bottom right

class SpotHeatMapDataset(ArrayBlockDataset):
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

            #! augmentation, not implemented

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

