import cv2
import torch
import numpy as np
import pandas as pd

from .dataset import *
from .util import *
from .draw import *

class BlockCornerCoordPredictor():
    def __init__(self, model, dataset:ArrayBlockDataset, device='cuda:0'):
        self.dataset = dataset
        self.device = device
        if isinstance(model, str):
            model = torch.load(model)
        self.model = model.to(device)
        self.model.eval()
        self.max_steps = 10

    def predict(self, img_path, gal, finetune=False, **finetune_args):
        '''
        pedict 3 xy coords of all blocks in an image
        return:
            coord_df: 3 xy coords of each block, original output from model
            spot_coord_df: xy coords of each spot, coords are finetuned if finetune is True
        '''
        idxs, xs, coords, scales = self.dataset.img2x(img_path, gal)
        if xs is None:
            return None

        with torch.no_grad():
            xs = torch.tensor(xs / 255).float().to(self.device)
            ypreds = self.model(xs).cpu().numpy()

        idxs = pd.MultiIndex.from_tuples(idxs).set_names(['img_path', 'Block', 'channel'])
        cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3']
        coord_df = pd.DataFrame(ypreds, index=idxs, columns=cols)

        # add window offset back to original coords
        for (img_path, b, c), block_coord_df in coord_df.groupby(['img_path', 'Block', 'channel']):
            w_start, w_end = get_window_coords(
                gal, b, expand=self.dataset.window_expand, pixel_size=self.dataset.pixel_size)
            coord_df.loc[block_coord_df.index, ['x1', 'x2', 'x3']] = block_coord_df[['x1', 'x2', 'x3']] / scales[b] + w_start[0]
            coord_df.loc[block_coord_df.index, ['y1', 'y2', 'y3']] = block_coord_df[['y1', 'y2', 'y3']] / scales[b] + w_start[1]
        spot_coord_df = self.to_spot_coords(coord_df, gal)

        if finetune:
            print('finetuning')
            return coord_df, self.finetune_coords(spot_coord_df, gal, img_path, finetune, **finetune_args)

        return coord_df, spot_coord_df

    def to_spot_coords(self, block_coord_df, gal):
        idxs, coords = [], []
        for (img_path, b, channel), df_row in block_coord_df.iterrows():
            n_cols = gal.header[f'Block{b}'][Gal.N_COLS]
            n_rows = gal.header[f'Block{b}'][Gal.N_ROWS]
            rstep = np.array([  # one step along row direction
                (df_row['x3'] - df_row['x2']) / (n_cols - 1),
                (df_row['y3'] - df_row['y2']) / (n_cols - 1)])
            cstep = np.array([  # one step along col direction
                (df_row['x2'] - df_row['x1']) / (n_rows - 1),
                (df_row['y2'] - df_row['y1']) / (n_rows - 1)])
            for c in range(1, n_cols+1):
                for r in range(1, n_rows+1):
                    top_left_spot = np.array([df_row['x1'], df_row['y1']])
                    spot_coord = top_left_spot + ((r-1) * cstep) + ((c-1) * rstep)
                    idxs.append((img_path, b, channel, c, r))
                    coords.append(spot_coord)
        idxs = pd.MultiIndex.from_tuples(idxs).set_names(['img_path', 'Block', 'Channel', 'Column', 'Row'])
        return pd.DataFrame(coords, index=idxs, columns=['x', 'y'])

    def finetune_coords(self, spot_coord_df, gal, img_path, finetune='all', **finetune_args):
        img = read_tif(img_path, channel_mode=self.dataset.channel_mode)
        for b, df in spot_coord_df.groupby(['Block']):
            p1, p2 = get_window_coords(
                gal, b, expand=self.dataset.window_expand, pixel_size=self.dataset.pixel_size)
            block_img = crop_window(img, p1, p2)
            coords = df[['x', 'y']].values - p1
            if (finetune == 'all') or (finetune == 'block'):
                print(f'finetuning block {b}')
                coords = self.finetune_block(block_img, coords, **finetune_args)
            if (finetune == 'all') or (finetune == 'spot'):
                print('finetuning spots')
                coords = self.finetune_spot(block_img, coords, **finetune_args)
            spot_coord_df.loc[df.index, ['x', 'y']] = coords + p1
        return spot_coord_df

    def finetune_block(self, img, spot_coords, mask_type='circle', mask_radius=5, stride=1):
        if mask_type == 'gaussian':
            mask = self.make_gaussian_masks(img.shape[-2:], spot_coords, mask_radius)
        elif mask_type == 'circle':
            mask = self.make_circle_masks(img.shape[-2:], spot_coords, mask_radius)
        else:
            raise ValueError('invalid mask type')

        directions = np.array([[0, -stride], [0, stride],
                               [-stride, 0], [stride, 0]])
        highest = 0
        for step in range(self.max_steps):
            scores = []
            for direction in directions:
                mask_ = self.shift_img(mask, direction)
                scores.append((mask_ * img).sum())
            max_idx = np.argmax(scores)
            if scores[max_idx] > highest:
                highest = scores[max_idx]
                mask = self.shift_img(mask, directions[max_idx])
                spot_coords += directions[max_idx]
            else:
                break
        print(f'{step} steps used')
        return spot_coords

    def finetune_spot(self, img, spot_coords, mask_type='circle', mask_radius=5, stride=1):
        directions = np.array([[0, -stride], [0, stride],
                            [-stride, 0], [stride, 0]])
        for i, coord in enumerate(spot_coords):
            if mask_type == 'gaussian':
                mask = self.make_gaussian_masks(img.shape[-2:], np.array([coord]), mask_radius)
            elif mask_type == 'circle':
                mask = self.make_circle_masks(img.shape[-2:], np.array([coord]), mask_radius)
            else:
                raise ValueError('invalid mask type')

            highest = 0
            for step in range(self.max_steps):
                scores = []
                for direction in directions:
                    mask_ = self.shift_img(mask, direction)
                    scores.append((mask_ * img).sum())
                max_idx = np.argmax(scores)
                if scores[max_idx] > highest:
                    highest = scores[max_idx]
                    mask = self.shift_img(mask, directions[max_idx])
                    spot_coords[i] += directions[max_idx]
                else:
                    break
        return spot_coords

    def make_gaussian_masks(self, img_shape, means, sigma):
        '''
        args:
            means: 2D array of shape (n, 2), x y coords of n spots
        returns:
            array of shape as img_shape
        '''
        xm, ym = means[:, 0], means[:, 1]
        n_spots = means.shape[0]
        xx, yy = np.meshgrid(np.arange(img_shape[1]),
                             np.arange(img_shape[0]))
        xx = np.moveaxis(np.stack([xx]*n_spots), 0, -1)
        yy = np.moveaxis(np.stack([yy]*n_spots), 0, -1)
        result = np.exp(-0.5 * (np.square(xx-xm) +
                                np.square(yy-ym)) / np.square(sigma))
        result = result.sum(axis=-1)
        result[result > (result.max() * .1)] = 0
        return result

    def make_circle_masks(self, img_shape, centers, radius):
        xm, ym = centers[:, 0], centers[:, 1]
        n_spots = centers.shape[0]
        xx, yy = np.meshgrid(np.arange(img_shape[1]),
                             np.arange(img_shape[0]))
        xx = np.moveaxis(np.stack([xx]*n_spots), 0, -1)
        yy = np.moveaxis(np.stack([yy]*n_spots), 0, -1)
        result = ((xx-xm)**2 + (yy-ym)**2) < radius**2
        return result.astype(np.uint8).sum(axis=-1)

    def shift_img(self, img, shift):
        mat = np.array([[1, 0, shift[0]], [0, 1, shift[1]]]).astype(np.float32)
        return cv2.warpAffine(img.astype(np.float32), mat, (img.shape[1], img.shape[0]))


class OldSchoolCoordPredictor():
    pass

if __name__ == '__main__':
    import IPython; IPython.embed(); exit()
