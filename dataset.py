import cv2
import os
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
from .draw import *


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


class XYbDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys

    def __getitem__(self, i):
        return self.xs[i], self.ys[i]

    def __len__(self):
        return len(self.xs)


class BlockSizeAugmentor():
    def __init__(self, block_size, cut_pp=None):
        '''
        args:
            block_size: original (n_rows, n_cols)
            cut_pp: ((l_cut_pp, r_cut_pp), (b_cut_pp, t_cut_pp))
        '''
        self.n_rows, self.n_cols = block_size
        # cut pp will be constant after the first augmentation
        if cut_pp is None:
            self.row_cut_pp, self.col_cut_pp = None, None
        else:
            self.row_cut_pp, self.col_cut_pp = cut_pp

    def make_cut_mask(self, img_shape, pts):
        # 2 pts [x, y] = [y, x] on img
        # d = (x−x1)(y2−y1)−(y−y1)(x2−x1)
        # sign(d) determines the side, left side of the line vec will be 0
        result = np.array([
            [0 if ((i-pts[0][1])*(pts[1][0]-pts[0][0]) -
                   (j-pts[0][0])*(pts[1][1]-pts[0][1]) < 0) else 1
                for j in range(img_shape[1])]
            for i in range(img_shape[0])])
        return result

    def cut_move_paste(self, img, cut_pts, shift):
        mask = self.make_cut_mask(img.shape, cut_pts)  # 0 / 1
        cut_img = mask * img
        remain_img = self.shift_img(1-mask, shift) * img
        return remain_img + self.shift_img(cut_img, shift)

    def augment_row(self, img, corners, inc):
        if inc == 0:
            return img, corners, None
        h_vec = corners[1] - corners[0]  # bottom left - top left
        h_step_vec = h_vec / (self.n_rows - 1)
        invalid_len = h_step_vec * abs(inc)
        valid_len = h_vec - 2*invalid_len
        if self.row_cut_pp is None:
            self.row_cut_pp = (np.random.rand(), np.random.rand())
        l_cut_coord = corners[0] + invalid_len + self.row_cut_pp[0] * valid_len
        r_cut_coord = corners[3] + invalid_len + self.row_cut_pp[1] * valid_len
        shift = h_step_vec * inc
        # left to right so that the moving part (bottom) of the mask is 1
        img_aug = self.cut_move_paste(img, (l_cut_coord, r_cut_coord), shift)
        corners[[1, 2]] += shift  # only the bottom 2 pts need to be shifted
        return img_aug, corners, np.array([l_cut_coord, r_cut_coord])

    def augment_col(self, img, corners, inc):
        if inc == 0:
            return img, corners, None
        w_vec = corners[3] - corners[0]  # top right - top left
        w_step_vec = w_vec / (self.n_cols - 1)
        invalid_len = w_step_vec * abs(inc)
        valid_len = w_vec - 2*invalid_len
        if self.col_cut_pp is None:
            self.col_cut_pp = (np.random.rand(), np.random.rand())
        t_cut_coord = corners[0] + invalid_len + self.col_cut_pp[0] * valid_len
        b_cut_coord = corners[1] + invalid_len + self.col_cut_pp[1] * valid_len
        shift = w_step_vec * inc
        # bottom to top so that the moving part (right side) of the mask is 1
        img_aug = self.cut_move_paste(img, (b_cut_coord, t_cut_coord), shift)
        corners[[2, 3]] += shift  # only the right 2 pts need to be shifted
        return img_aug, corners, np.array([b_cut_coord, t_cut_coord])

    def augment(self, img, corners, row_inc=0, col_inc=0):
        '''
        args:
            corners: [top left, bottom left, bottom right, top right] (4, 2)
            row_inc (int): changes n_row
        returns:
            aug_img
            coords: corner coords (4, 2)
        '''
        img_aug, corners_aug,  row_cut_coords = self.augment_row(
            img, corners, row_inc)
        img_aug, corners_aug, col_cut_coords = self.augment_col(
            img_aug, corners_aug, col_inc)
        kpts = KeypointsOnImage([
            Keypoint(x=corners_aug[0, 0],
                     y=corners_aug[0, 1]),
            Keypoint(x=corners_aug[1, 0],
                     y=corners_aug[1, 1]),
            Keypoint(x=corners_aug[2, 0],
                     y=corners_aug[2, 1]),
            Keypoint(x=corners_aug[3, 0],
                     y=corners_aug[3, 1]),
        ], shape=img_aug.shape)
        return img_aug, kpts, (row_cut_coords, col_cut_coords)

    def shift_img(self, img, shift):
        mat = np.array([[1, 0, shift[0]], [0, 1, shift[1]]]).astype(np.float32)
        return cv2.warpAffine(img.astype(np.float32), mat, (img.shape[1], img.shape[0]))


class ArrayBlockDataset():
    def __init__(self, window_resize, window_expand, equalize,
                 morphology, pixel_size, clahe_kwargs={'clipLimit': 20, 'tileGridSize': (8, 8)}):
        self.window_resize = window_resize
        self.window_expand = window_expand
        self.equalize = equalize
        self.clahe_kwargs = clahe_kwargs
        self.morphology = morphology
        self.pixel_size = pixel_size
        # self.block_size_augmentor = BlockSizeAugmentor()

    def check_out_of_bounds(self, img_shape, coord):
        h, w = img_shape
        return (coord[:, 0].max() >= w) or (coord[:, 1].max() >= h) or (coord[:, 0].min() < 0) or (coord[:, 1].min() < 0)

    def block_size_augmentation(img, coord_df):
        pass
        # return aug_img, aug_coord_df

    def img2x(self, img_path, gal, gpr=None):
        '''
        Single img file to xs for all blocks and fused channels
        args:
            img_info: parsed json file, must contain BlockCoords

        returns:
            xs:
                cropped window, grayscale[0~255]=>[0~1]
                shape ( #blocks, 1, win_shape[0], win_shape[1] )
            coords:
                list of df, each spot coord in the block
            idxs:
                (img_path, block) for one x sample
            scales (dict):
                img scaling for each block
        '''
        ims = read_tif(img_path)
        if ims:
            im = np.stack(ims).max(axis=0)  # fuse img channels
        else:
            return None, None, None, None

        xs, coords, idxs, scales = [], [], [], dict()
        for b in range(1, gal.header['BlockCount']+1):
            idxs.append((img_path, b))
            p1, p2 = get_window_coords(
                gal, b, self.window_expand, pixel_size=self.pixel_size)
            cropped = crop_window(im, p1, p2).astype('uint8')
            cropped, scale = im_contain_resize(cropped, self.window_resize)
            scales[b] = scale
            if self.equalize:
                cropped = im_equalize(
                    cropped, method=self.equalize, clahe_kwargs=self.clahe_kwargs)
            if self.morphology:
                cropped = im_morphology(cropped)
            xs.append(cropped)

            if gpr is not None:
                coord_df = (
                    gpr.data.loc[b][['X', 'Y']] / self.pixel_size - np.array(p1)) * scale
                coords.append(coord_df)

        xs = np.stack(xs)
        return idxs, np.expand_dims(xs, axis=1)/255, coords, scales

    def read_dir_files(self, dirs, gal_paths, exclude=[]):
        results = []
        for d, g in zip(dirs, gal_paths):
            for f in os.listdir(d):
                f = os.path.join(d, f)
                if f.endswith('.tif') and not (f in exclude):
                    results.append((f, g, f.replace('.tif', '.gpr')))
        return results

    def get_dataloaders(self, dirs, gal_paths, te_size=.2, exclude=[], batch_size=1, save_dir=None):
        '''
        args:
            exclude: files that is corrupted or preserved for independent test
        returns: (trainDataset, valDataset, testDataset)
        '''
        # each directory have its own tr te split
        ims_tr, gals_tr, gprs_tr = [], [], []
        ims_te, gals_te, gprs_te = [], [], []
        for d, g in zip(dirs, gal_paths):
            ims, gals, gprs = [], [], []
            for f in os.listdir(d):
                f = os.path.join(d, f)
                if f.endswith('.tif') and not (f in exclude):
                    ims.append(f)
                    gprs.append(f.replace('.tif', '.gpr'))
                    gals.append(g)
            imtr, imte, galtr, galte, gprtr, gprte = train_test_split(
                ims, gals, gprs, test_size=te_size, shuffle=True)
            ims_tr.extend(imtr)
            ims_te.extend(imte)
            gals_tr.extend(galtr)
            gals_te.extend(galte)
            gprs_tr.extend(gprtr)
            gprs_te.extend(gprte)
        if save_dir:
            write_file(list(zip(ims_tr, gals_tr, gprs_tr)),
                       os.path.join(save_dir, 'tr.txt'))
            write_file(list(zip(ims_te, gals_te, gprs_te)),
                       os.path.join(save_dir, 'te.txt'))
        return (
            DataLoader(ImgGalGprDataset(ims_tr, gals_tr, gprs_tr),batch_size=batch_size, shuffle=True),
            DataLoader(ImgGalGprDataset(ims_te, gals_te, gprs_te), batch_size=batch_size, shuffle=True))


class BlockCornerCoordDataset(ArrayBlockDataset):
    def __init__(self, window_resize=None, window_expand=2, equalize=False, morphology=False, pixel_size=10):
        super().__init__(window_resize=window_resize, window_expand=window_expand,
                         equalize=equalize, morphology=morphology, pixel_size=pixel_size)
        self.aug_seq = aug.Sequential([
            aug.HorizontalFlip(0.5),
            aug.VerticalFlip(0.5),
            aug.Sometimes(0.5, aug.GaussianBlur(sigma=(0, 0.2))),
            aug.LinearContrast((0.8, 1.2)),
            aug.AdditiveGaussianNoise(scale=(0.0, 0.05*255)),
            aug.Multiply((0.5, 1.0)),
            aug.Affine(
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-5, 5),
                scale=(.9, 1.1)
            )], random_order=True)
        self.cache = dict()

    def img2xy(self, img_path, gal_path, gpr_path, augment=0,
               bsa_args: dict = None, blocks=None, keep_ori=True):
        '''
        Single img file to xys for all blocks and channels
        args:
            blocks:
                If None, get all blocks in img.
                If given list / set, get only the blocks in the list
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

        gal = make_fake_gal(Gpr(gpr_path)) if gal_path == '' else Gal(gal_path)
        idxs, imgs, df, scales = super().img2x(
            img_path, gal, Gpr(gpr_path))  # imgs: (N, 1, h, w)
        if (imgs is None) or (df is None):
            return None, None
        h, w = imgs.shape[2], imgs.shape[3]
        xs, ys = [], []
        for (img_path, b), img, block_df in zip(idxs, imgs, df):
            if (blocks is not None) and not (b in blocks):
                continue
            nrows = gal.header[f'Block{b}'][Gal.N_ROWS]
            ncols = gal.header[f'Block{b}'][Gal.N_COLS]
            kpts = KeypointsOnImage([
                Keypoint(x=block_df.loc[1, 1]['X'],
                         y=block_df.loc[1, 1]['Y']),
                Keypoint(x=block_df.loc[nrows, 1]['X'],
                         y=block_df.loc[nrows, 1]['Y']),
                Keypoint(x=block_df.loc[nrows, ncols]['X'],
                         y=block_df.loc[nrows, ncols]['Y']),
                Keypoint(x=block_df.loc[1, ncols]['X'],
                         y=block_df.loc[1, ncols]['Y']),
            ], shape=(h, w))

            if keep_ori:
                xs.append(img)
                coord = self.to_Lcoord(kpts.to_xy_array())
                ys.append(coord.flatten())

            if bsa_args is not None:
                bsa = BlockSizeAugmentor((nrows, ncols))
                if bsa_args['n'] >= 1:
                    augxs, augys = [], []
                    while len(augxs) < bsa_args['n']:
                        row_inc = np.random.choice(bsa_args['row_inc_choice'])
                        col_inc = np.random.choice(bsa_args['col_inc_choice'])
                        img_aug, kpts_aug, cut_coords = bsa.augment(
                            img[0], kpts.to_xy_array(), row_inc, col_inc)
                        coord = self.to_Lcoord(
                            kpts_aug.to_xy_array())  # (3, 2)
                        if self.check_out_of_bounds(img_aug.shape, coord):
                            continue  # skip if coord out of bounds
                        augxs.append(np.array([img_aug]))
                        augys.append(coord.flatten())
                    xs.extend(augxs)
                    ys.extend(augys)
                elif np.random.rand() < bsa_args['n']:
                    row_inc = np.random.choice(bsa_args['row_inc_choice'])
                    col_inc = np.random.choice(bsa_args['col_inc_choice'])
                    img_aug, kpts_aug, cut_coords = bsa.augment(
                        img[0], kpts.to_xy_array(), row_inc, col_inc)
                    coord = self.to_Lcoord(kpts_aug.to_xy_array())  # (3, 2)
                    if self.check_out_of_bounds(img_aug.shape, coord):
                        continue  # skip if coord out of bounds
                    xs.append(np.array([img_aug]))
                    ys.append(coord.flatten())

            if augment >= 1:
                augxs, augys = [], []
                while len(augxs) < augment:
                    img_aug, kpts_aug = self.aug_seq(
                        image=(img[0]*255).astype('uint8'), keypoints=kpts)  # img: (1, w, h) -> (w, h)
                    coord = self.to_Lcoord(kpts_aug.to_xy_array())  # (3, 2)
                    if self.check_out_of_bounds(img_aug.shape, coord):
                        continue  # skip if coord out of bounds
                    augxs.append(np.array([img_aug/255]))
                    augys.append(coord.flatten())
                xs.extend(augxs)
                ys.extend(augys)
            elif np.random.rand() < augment:
                img_aug, kpts_aug = self.aug_seq(
                    image=(img[0]*255).astype('uint8'), keypoints=kpts)  # img: (1, w, h) -> (w, h)
                coord = self.to_Lcoord(kpts_aug.to_xy_array())  # (3, 2)
                if self.check_out_of_bounds(img_aug.shape, coord):
                    continue  # skip if coord out of bounds
                xs.append(np.array([img_aug/255]))
                ys.append(coord.flatten())

        return np.stack(xs), np.stack(ys)

    def imgs2xy(self, data, augment=0, bsa_args=None, keep_ori=True):
        '''
        List of img files to xy for all blocks
        data: [(img, gal, gpr), (...)]
        returns: aggregated x, y nparrays
        '''
        xs, ys = [], []
        for img_path, gal_path, gpr_path in data:
            x, y = self.img2xy(
                img_path, gal_path, gpr_path, augment, bsa_args, keep_ori=keep_ori)
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
        coord = coord[np.argsort(coord[:, 0])]  # sort by x
        # left 2 points sort by y
        coord[:2] = coord[:2][np.argsort(coord[:2][:, 1])]
        # right 2 points sort by y
        coord[2:] = coord[2:][np.argsort(coord[2:][:, 1])]
        return coord[[0, 1, 3]]  # top left, bottom left, bottom right


class MetaBlockCornerCoordDataset(ArrayBlockDataset):
    def __init__(self, window_resize=None, window_expand=2, equalize=False, morphology=False, pixel_size=10,
                 dataset_choices:dict=None, row_inc_choices=None, col_inc_choices=None, scale_choices=None,
                 h_flip=None, v_flip=None, blur_choices=None, noise_choices=None):
        super().__init__(window_resize=window_resize, window_expand=window_expand,
                         equalize=equalize, morphology=morphology, pixel_size=pixel_size)
        self.dataset_choices = dataset_choices  # {dataset_name: ([img_dir], [gal_path])}
        self.row_inc_choices = row_inc_choices  # number
        self.col_inc_choices = col_inc_choices
        self.scale_choices = scale_choices  # number
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.blur_choices = blur_choices
        self.noise_choices = noise_choices

        self.aug_seq = aug.Sequential([
            aug.Affine(
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-5, 5),
            )])

        self.img_cache = dict()  # saved img2x
        self.gal_cache = dict()  # saved gals, key=dataset name

    def get_task(self, from_n_imgs=None, augment=0):
        ds_name = np.random.choice(list(self.dataset_choices.keys()))
        img_dir, gal_path = self.dataset_choices[ds_name]
        row_inc = np.random.choice(
            self.row_inc_choices) if self.row_inc_choices is not None else 0
        col_inc = np.random.choice(
            self.col_inc_choices) if self.col_inc_choices is not None else 0
        scale = np.random.choice(
            self.scale_choices) if self.col_inc_choices is not None else 0
        blur = np.random.choice(
            self.blur_choices) if self.blur_choices is not None else 0
        noise = np.random.choice(
            self.noise_choices) if self.noise_choices is not None else 0
        h_flip = True if self.h_flip and (np.random.rand() < self.h_flip) else False
        v_flip = True if self.v_flip and (np.random.rand() < self.v_flip) else False

        data = self.read_dir_files(img_dir, gal_path)
        if from_n_imgs is not None:
            idxs = np.random.randint(0, len(data), size=from_n_imgs).astype(int)
            data = np.array(data)[idxs]
        args = {
            'dataset': ds_name,
            'gal': gal_path,
            'samples': list(data[:, 0]),
            'bsa': [int(row_inc), int(col_inc)],
            'scale': scale,
            'blur': blur,
            'noise': noise,
            'h_flip': h_flip,
            'v_flip': v_flip,
        }
        xs, ys = self.imgs2xy(ds_name, data, row_inc, col_inc, scale, h_flip, v_flip,
            blur, noise, augment=augment)
        if xs is None:
            return None, args
        idxs = [i for i in range(len(xs))]
        np.random.shuffle(idxs)  # inplace
        xs, ys = xs[idxs], ys[idxs]  # shuffle

        return (xs, ys), args

    def img2xy(self, ds_name, img_path, gal_path, gpr_path,
            row_inc, col_inc, scale, h_flip, v_flip, blur, noise,
            augment=0, cut_pp=None):

        if ds_name in self.gal_cache:
            gal = self.gal_cache[ds_name]
        else:
            gal = make_fake_gal(Gpr(gpr_path)) if gal_path == '' else Gal(gal_path)
            self.gal_cache[ds_name] = gal

        if img_path in self.img_cache:
            idxs, imgs, df, scales = self.img_cache[img_path]
        else:
            idxs, imgs, df, scales = super().img2x(
                img_path, gal, Gpr(gpr_path))  # imgs: (N, 1, h, w)
            self.img_cache[img_path] = idxs, imgs, df, scales
        if (imgs is None) or (df is None):
            print(f'{img_path} failed.')
            return None, None, None

        h, w = imgs.shape[2], imgs.shape[3]
        xs, ys = [], []
        for (img_path, b), img, block_df in zip(idxs, imgs, df):
            nrows = gal.header[f'Block{b}'][Gal.N_ROWS]
            ncols = gal.header[f'Block{b}'][Gal.N_COLS]
            kpts = KeypointsOnImage([
                Keypoint(x=block_df.loc[1, 1]['X'],
                         y=block_df.loc[1, 1]['Y']),
                Keypoint(x=block_df.loc[nrows, 1]['X'],
                         y=block_df.loc[nrows, 1]['Y']),
                Keypoint(x=block_df.loc[nrows, ncols]['X'],
                         y=block_df.loc[nrows, ncols]['Y']),
                Keypoint(x=block_df.loc[1, ncols]['X'],
                         y=block_df.loc[1, ncols]['Y']),
            ], shape=(h, w))

            # bsa
            bsa = BlockSizeAugmentor((nrows, ncols), cut_pp)
            img, kpts, _ = bsa.augment(
                img[0], kpts.to_xy_array(), row_inc, col_inc)
            coord = self.to_Lcoord(kpts.to_xy_array())
            if self.check_out_of_bounds(img.shape, coord):
                continue

            img = (img * 255).astype('uint8')
            # h_flip
            if h_flip:
                aug_func = aug.HorizontalFlip(1)
                img, kpts = aug_func(image=img, keypoints=kpts)
            # v_flip
            if v_flip:
                aug_func = aug.VerticalFlip(1)
                img, kpts = aug_func(image=img, keypoints=kpts)
            # scale
            if scale != 1:
                aug_func = aug.Affine(scale=scale)
                img, kpts = aug_func(image=img, keypoints=kpts)
            # blur
            if blur != 0:
                aug_func = aug.GaussianBlur(sigma=blur)
                img, kpts = aug_func(image=img, keypoints=kpts)
            # noise
            if noise != 0:
                aug_func = aug.AdditiveGaussianNoise(scale=noise)
                img, kpts = aug_func(image=img, keypoints=kpts)
            img = img / 255

            # augment
            if augment >= 1:
                augxs, augys = [], []
                while len(augxs) < augment:
                    img_aug, kpts_aug = self.aug_seq(
                        image=(img*255).astype('uint8'), keypoints=kpts)  # img: (1, w, h) -> (w, h)
                    coord = self.to_Lcoord(kpts_aug.to_xy_array())  # (3, 2)
                    if self.check_out_of_bounds(img_aug.shape, coord):
                        continue  # skip if coord out of bounds
                    augxs.append(np.array([img_aug/255]))
                    augys.append(coord.flatten())
                xs.extend(augxs)
                ys.extend(augys)
            elif np.random.rand() < augment:
                img_aug, kpts_aug = self.aug_seq(
                    image=(img*255).astype('uint8'), keypoints=kpts)  # img: (1, w, h) -> (w, h)
                coord = self.to_Lcoord(kpts_aug.to_xy_array())  # (3, 2)
                if self.check_out_of_bounds(img_aug.shape, coord):
                    continue  # skip if coord out of bounds
                xs.append(np.array([img_aug/255]))
                ys.append(coord.flatten())

            coord = self.to_Lcoord(kpts.to_xy_array())
            if self.check_out_of_bounds(img.shape, coord):
                continue
            xs.append(np.array([img]))
            ys.append(coord.flatten())

        if len(xs) <= 0:
            return None, None, None
        return np.stack(xs), np.stack(ys), (bsa.row_cut_pp, bsa.col_cut_pp)

    def imgs2xy(self, ds_name, data,
            row_inc, col_inc, scale, h_flip, v_flip, blur, noise,
            augment=0, cut_pp=None):
        '''
        List of img files to xy for all blocks
        args: lists of img and gpr paths
        returns: aggregated x, y nparrays
        '''
        xs, ys = [], []
        for img_path, gal_path, gpr_path in data:
            x, y, cut_pp = self.img2xy(
                ds_name, img_path, gal_path, gpr_path,
                row_inc, col_inc, scale, h_flip, v_flip, blur, noise, augment, cut_pp)
            if (x is not None) and (y is not None):
                xs.append(x)
                ys.append(y)
        if len(xs) <= 0:
            return None, None
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    def to_Lcoord(self, coord):
        '''
        coord: (4, 2) not sorted -> (3, 2) sorted L coord
        '''
        coord = coord[np.argsort(coord[:, 0])]  # sort by x
        # left 2 points sort by y
        coord[:2] = coord[:2][np.argsort(coord[:2][:, 1])]
        # right 2 points sort by y
        coord[2:] = coord[2:][np.argsort(coord[2:][:, 1])]
        return coord[[0, 1, 3]]  # top left, bottom left, bottom right


class BlockCornerHeatmapDataset(ArrayBlockDataset):
    def __init__(self, window_size=None, window_expand=2, down_sample=4,
                 equalize=False, morphology=False, pixel_size=10, sigma=1):
        super().__init__(window_size=window_size, window_expand=window_expand, down_sample=down_sample,
                         equalize=equalize, morphology=morphology, pixel_size=pixel_size)

        self.sigma = sigma
        self.aug_seq = aug.Sequential([
            aug.HorizontalFlip(0.5),
            aug.VerticalFlip(0.5),
            aug.Sometimes(0.5, aug.GaussianBlur(sigma=(0, 0.2))),
            aug.LinearContrast((0.8, 1.2)),
            aug.AdditiveGaussianNoise(scale=(0.0, 0.05*255)),
            aug.Multiply((0.5, 1.0)),
            aug.Affine(
                scale=(0.7, 1),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-5, 5)
            )], random_order=True)

    def img2xy(self, img_path, gal, gpr, augment=0):
        '''
        Single img file to xys for all blocks and channels
        returns:
            xs:
                cropped window, grayscale[0~255]=>[0~1]
                shape ( #blocks, 1, win_shape[0], win_shape[1] )
            ys:
                gaussian heatmaps of same w, h as xs, one channel for one kpt
            idx:
                (img_path, block) for one x sample
        '''
        idxs, imgs, df = super().img2x(img_path, gal, gpr)  # imgs: (N, 1, h, w)
        h, w = imgs.shape[2], imgs.shape[3]
        gpr = Gpr(gpr) if type(gpr) == str else gpr
        xs, ys, coords = [], [], []
        for (img_path, b), img, block_df in zip(idxs, imgs, df):
            nrows = gal.header[f'Block{b}'][Gal.N_ROWS]
            ncols = gal.header[f'Block{b}'][Gal.N_COLS]
            kpts = KeypointsOnImage([
                Keypoint(x=block_df.loc[1, 1]['X'], y=block_df.loc[1, 1]['Y']),
                Keypoint(x=block_df.loc[1, ncols]['X'],
                         y=block_df.loc[1, ncols]['Y']),
                Keypoint(x=block_df.loc[nrows, ncols]['X'],
                         y=block_df.loc[nrows, ncols]['Y']),
                Keypoint(x=block_df.loc[nrows, 1]['X'],
                         y=block_df.loc[nrows, 1]['Y'])
            ], shape=(h, w))

            if augment <= 0:
                xs.append(img)
                coord = self.to_Lcoord(kpts.to_xy_array())
                ys.append(self.coord2heatmap(
                    coord, (img.shape[-2], img.shape[-1])))
                coords.append(coord.flatten())
            else:
                for i in range(augment):
                    img_aug, kpts_aug = self.aug_seq(
                        image=(img[0]*255).astype('uint8'), keypoints=kpts)  # img: (1, w, h) -> (w, h)
                    coord = self.to_Lcoord(kpts_aug.to_xy_array())  # (3, 2)
                    if self.check_out_of_bounds(img_aug.shape, coord):
                        continue  # skip if coord out of bounds
                    xs.append(np.array([img_aug/255]))
                    ys.append(self.coord2heatmap(
                        coord, (img_aug.shape[-2], img_aug.shape[-1])))
                    coords.append(coord.flatten())

        return np.stack(xs), np.stack(ys), np.stack(coords)

    def imgs2xy(self, img_paths, gal_paths, gpr_paths, augment=True):
        '''
        List of img files to xy for all blocks
        args: lists of img and gpr paths
        returns: aggregated x, y nparrays
        '''
        xs, ys, coords = [], [], []
        for img_path, gal_path, gpr_path in zip(img_paths, gal_paths, gpr_paths):
            x, y, coord = self.img2xy(
                img_path, Gal(gal_path), gpr_path, augment=augment)
            if (x is not None) and (y is not None):
                xs.append(x)
                ys.append(y)
                coords.append(coord)
        # pass if all cropped images are of same shape
        xs, ys, coords = np.concatenate(xs, axis=0), np.concatenate(
            ys, axis=0), np.concatenate(coords, axis=0)
        return xs, ys, coords

    def to_Lcoord(self, coord):
        '''
        coord: (4, 2) not sorted -> (3, 2) sorted L coord
        '''
        coord = coord[np.argsort(coord[:, 0])]  # sort by x
        # left 2 points sort by y
        coord[:2] = coord[:2][np.argsort(coord[:2][:, 1])]
        # right 2 points sort by y
        coord[2:] = coord[2:][np.argsort(coord[2:][:, 1])]
        return coord[[0, 1, 3]]  # top left, bottom left, bottom right

    def coord2heatmap(self, coords, img_shape):
        '''
        coords: (n, 2) for n spots in one block
        '''
        result = [self.make_2D_gaussian(
            img_shape, coord, self.sigma) for coord in coords]
        return np.stack(result, axis=0)

    def make_2D_gaussian(self, img_shape, mean, sigma):
        xm, ym = mean
        xx, yy = np.meshgrid(
            np.arange(img_shape[1]),
            np.arange(img_shape[0]))
        result = np.exp(-0.5 * (np.square(xx-xm) +
                                np.square(yy-ym)) / np.square(sigma))
        return result / np.max(result)


def get_dataset(data_dir, gal, args, save_dir=None):
    '''
    data_dir: list of str paths
    '''
    dataset = BlockCornerCoordDataset(
        window_resize=args['window_resize'],
        window_expand=args['window_expand'],
        equalize=args['equalize'], morphology=args['morphology'])
    data = dataset.read_dir_files(data_dir, gal)
    xs, ys = dataset.imgs2xy(data, augment=args['augment'], keep_ori=True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            f.write(json.dumps(args))
        np.save(os.path.join(save_dir, 'x.npy'), xs)
        np.save(os.path.join(save_dir, 'y.npy'), ys)


def write_datasets():
    data_dirs = [
        ['gridding/data/DeRisi/train']]
    gal_dirs = [
        ['']]
    args = {
        'data_dirs': data_dirs,
        'gal_dirs': gal_dirs,
        'window_resize': (256, 256),
        'window_expand': 2,
        'equalize': False,
        'morphology': False,
        'augment': 0,
    }
    output_dir = 'gridding/data/npy/'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        f.write(json.dumps(args))
    dataset = BlockCornerCoordDataset(
        window_resize=args['window_resize'],
        window_expand=args['window_expand'],
        equalize=args['equalize'], morphology=args['morphology'])
    for data_dir, gal in zip(data_dirs, gal_dirs):
        data = dataset.read_dir_files(data_dir, gal)
        xs, ys = dataset.imgs2xy(data, augment=args['augment'], keep_ori=True)
        print(data_dir)
        print(xs.shape, ys.shape)
        out = os.path.join(output_dir, data_dir[0].split('/')[2])
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, 'x.npy'), xs)
        np.save(os.path.join(out, 'y.npy'), ys)


def write_reptile_dataset():
    pass


if '__main__' == __name__:
    dataset = BlockCornerCoordDataset(
        window_resize=(256, 256), window_expand=2,
        equalize=False, morphology=False)
    xb, yb = dataset.img2xy(
        'gridding/data/GEO/GSM15898_CH2.tif',
        make_fake_gal(Gpr('gridding/data/GEO/GSM15898_CH2.gpr')),
        'gridding/data/GEO/GSM15898_CH2.gpr', augment=0)
    idxs = np.random.randint(xb.shape[0], size=5)
    for i, (x, y) in enumerate(zip(xb[idxs], yb[idxs])):
        im = x2rgbimg(x)
        im = draw_parallelogram(im, y.reshape(3, 2), color=(0, 255, 0))
        cv2.imwrite(f'gridding/imgs/test/GEO_{i}.png', im)
