import cv2
import torch
import numpy as np
import pandas as pd

from .dataset import *
from .util import *
from .draw import *

class BlockCornerCoordPredictor():
    def __init__(self, model_path, window_expand=2, down_sample=4, pixel_size=10,
            equalize=None, morphology=False, device='cuda:1'):
        self.dataset = BlockCornerCoordDataset(window_expand, down_sample, equalize, morphology, pixel_size)
        self.window_expand = window_expand
        self.down_sample = down_sample
        self.pixel_size = pixel_size
        self.equalize = equalize
        self.morphology = morphology
        self.device = device
        self.model = torch.load(model_path, map_location=device).eval()
        print('model loaded')

    def predict(self, img_path, gal, finetune=False, spot_radius=5):
        '''
        pedict 3 xy coords of all blocks in an image
        return:
            ypreds: coords respective to window coords
            idxs: (img_path, block_no, channel)
        '''
        gal = Gal(gal) if type(gal) == str else gal
        xs, idxs = self.dataset.img2x(img_path, gal)

        with torch.no_grad():
            xs = torch.tensor(xs).float().to(self.device)
            ypreds = self.model(xs).cpu().numpy()

        idxs = pd.MultiIndex.from_tuples(idxs).set_names(['img_path', 'block'])
        cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3']
        coord_df = pd.DataFrame(ypreds, index=idxs, columns=cols) * self.down_sample

        # add window offset back to original coords
        for (img_path, b), block_coord_df in coord_df.groupby(['img_path', 'block']):
            w_start, w_end = get_window_coords(b, gal, self.window_expand, pixel_size=self.pixel_size)
            coord_df.loc[block_coord_df.index, ['x1', 'x2', 'x3']] = block_coord_df[['x1', 'x2', 'x3']] + w_start[0]
            coord_df.loc[block_coord_df.index, ['y1', 'y2', 'y3']] = block_coord_df[['y1', 'y2', 'y3']] + w_start[1]


        if finetune:
            return self.finetune_spot_coords(coord_df, gal, img_path, spot_radius)
        else:
            return self.to_spot_coords(coord_df, gal)

    def finetune_spot_coords(self, coord_df, gal, img_path, spot_radius):
        img = cv2.imread(img_path, 0)
        spot_coord_df = self.to_spot_coords(coord_df, gal)
        spot_xs = spot_coord_df['x']
        spot_ys = spot_coord_df['y']
        spot_coords = list(zip(spot_xs, spot_ys))
        spot_coords = self.finetune_whole_img(img, spot_coords, spot_radius)
        spot_coords = self.finetune_per_spot(img, spot_coords, spot_radius)
        finetune_xs, finetune_ys = list(zip(*spot_coords))
        spot_coord_df['x'] = finetune_xs
        spot_coord_df['y'] = finetune_ys
        return spot_coord_df

    def finetune_whole_img(self, img, spot_coords, spot_radius):
        stop = False
        while not stop:
            four_direction_centers = []
            four_direction_brightness = np.zeros(4)
            curr_brightness = 0
            for i, (x, y) in enumerate(spot_coords):
                curr_four_direction_centers = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
                four_direction_centers.append(curr_four_direction_centers)
                four_direction_brightness += self.get_brightness(img, curr_four_direction_centers, radius=spot_radius)
                curr_brightness += self.get_brightness(img, [(x, y)], radius=spot_radius)[0]

            if max(four_direction_brightness) > curr_brightness:
                spot_coords = list(zip(*four_direction_centers))[np.argmax(four_direction_brightness)]
            else:
                stop = True
        return list(spot_coords)

    def finetune_per_spot(self, img, spot_coords, spot_radius):
        for i in range(len(spot_coords)):
            brighter = True
            tmp = []
            while brighter:
                brighter = False
                x, y = spot_coords[i]
                forward_direction_centers = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
                if tmp:
                    forward_direction_centers.remove(tmp)
                forward_direction_brightness = self.get_brightness(img, forward_direction_centers, radius=spot_radius)
                curr_brightness = self.get_brightness(img, [(x, y)], radius=spot_radius)[0]
                tmp = spot_coords[i]

                if max(forward_direction_brightness) > curr_brightness:
                    spot_coords[i] = forward_direction_centers[np.argmax(forward_direction_brightness)]
                    brighter = True
        return spot_coords

    def create_mask(self, radius):
        row, col = radius*2+1, radius*2+1
        center_row, center_col = radius, radius
        Y, X = np.ogrid[: row, : col]
        dist_from_center = np.sqrt((Y-center_row)**2 + (X-center_col)**2)
        mask = dist_from_center <= radius
        return mask.astype('int')

    def get_brightness(self, img, centers, radius):
        brightness = []
        mask = self.create_mask(radius=radius)
        for center in centers:
            x, y = int(round(center[0])), int(round(center[1]))
            crop = img[y-radius: y+radius+1, x-radius: x+radius+1]
            b = (crop * mask).sum()
            brightness.append(b)
        return np.array(brightness)

    def to_spot_coords(self, block_coord_df, gal):
        idxs, coords = [], []
        for (img_path, b), df_row in block_coord_df.iterrows():
            n_cols = gal.header[f'Block{b}'][Gal.N_COLS]
            n_rows = gal.header[f'Block{b}'][Gal.N_ROWS]
            rstep = np.array([ # one step along row direction
                (df_row['x3'] - df_row['x2']) / (n_cols - 1),
                (df_row['y3'] - df_row['y2']) / (n_cols - 1)])
            cstep = np.array([ # one step along col direction
                (df_row['x2'] - df_row['x1']) / (n_rows - 1),
                (df_row['y2'] - df_row['y1']) / (n_rows - 1)])
            for c in range(1, n_cols+1):
                for r in range(1, n_rows+1):
                    top_left_spot = np.array([df_row['x1'], df_row['y1']])
                    spot_coord = top_left_spot + ((r-1) * cstep) + ((c-1) * rstep)
                    idxs.append((img_path, b, c, r))
                    coords.append(spot_coord)
        idxs = pd.MultiIndex.from_tuples(idxs).set_names(['img_path', 'block', 'col', 'row'])
        return pd.DataFrame(coords, index=idxs, columns=['x', 'y'])

class OldSchoolCoordPredictor():
    pass

if __name__ == '__main__':
    import IPython; IPython.embed(); exit()
