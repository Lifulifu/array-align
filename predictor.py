import torch
import pandas as pd

class ResnetPredictor():
    def __init__(self, model_path, window_expand=2, down_sample=4,
        equalize='clahe', morphology=True, device='cuda:0'):

        self.window_expand = window_expand
        self.down_sample = down_sample
        self.equalize = equalize
        self.morphology = morphology

        self.device = device
        self.model = torch.load(model_path).eval().to(device)
        print('model loaded')

    def predict_block_coords(self, img_path, gal_file):
        '''
        pedict 3 xy coords of all blocks in an image
        return:
            ypreds: coords respective to window coords
            ypreds_ori: transform into big image pixel coords
            idxs: (img_path, block_no, channel)
        '''
        gal = Gal(gal_file) if type(gal_file) == str else gal_file
        xs, idxs = img2x(img_path, gal,
            window_expand=self.window_expand,
            down_sample=self.down_sample,
            equalize=self.equalize,
            morphology=self.morphology)

        with torch.no_grad():
            xs = torch.tensor(xs).float().to(self.device)
            ypreds = self.model(xs).cpu().numpy()

        idxs = pd.MultiIndex.from_tuples(idxs).set_names(['img_path', 'block'])
        cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3']
        coord_df = pd.DataFrame(ypreds, index=idxs, columns=cols)

        # add window offset and downsample back to original coords
        for (img_path, b), block_coord_df in coord_df.groupby(['img_path', 'block']):
            w_start, w_end = get_window_coords(b, gal, self.window_expand)
            coord_df.loc[block_coord_df.index, ['x1', 'x2', 'x3']] = block_coord_df[['x1', 'x2', 'x3']] * self.down_sample + w_start[0]
            coord_df.loc[block_coord_df.index, ['y1', 'y2', 'y3']] = block_coord_df[['y1', 'y2', 'y3']] * self.down_sample + w_start[1]
        return coord_df

    def to_spot_coords(self, block_coord_df, gal_file):
        gal = Gal(gal_file) if type(gal_file) == str else gal_file
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

class OldSchoolPredictor():
    pass

if __name__ == '__main__':
    import IPython; IPython.embed(); exit()