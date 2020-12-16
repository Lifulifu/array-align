import torch
import pandas as pd

from .dataset import *
from .util import *
from .draw import *

class BlockCornerCoordPredictor():
    def __init__(self, model_path, window_expand=2, down_sample=4, pixel_size=10,
            equalize=None, morphology=False, device='cuda:0'):
        self.dataset = BlockCornerCoordDataset(window_expand, down_sample, equalize, morphology, pixel_size)
        self.window_expand = window_expand
        self.down_sample = down_sample
        self.pixel_size = pixel_size
        self.equalize = equalize
        self.morphology = morphology
        self.device = device
        self.model = torch.load(model_path, map_location=device).eval()
        print('model loaded')

    def predict(self, img_path, gal):
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
        return self.to_spot_coords(coord_df, gal)

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