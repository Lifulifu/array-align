import cv2
import os
import numpy as np
from typing import List
import blend_modes

from .util import *

def to_color(grayscale, channel, alpha=False):  # {{{
    """
    Arguments:
        color(str): the color the grayscale will be converted to
        grayscale(array): two dimension gray image

    Returns:
        rgb(array): three dimension image and only red channel or green channel
    """
    if alpha:
        rgb = np.zeros((*grayscale.shape, 4)).astype(np.uint8)
        rgb[:, :, 3] = 255
    else:
        rgb = np.zeros((*grayscale.shape, 3)).astype(np.uint8)
    rgb[:, :, channel] = grayscale
    return rgb

def x2rgbimg(x, alpha=False, eq=False):
    '''
    x is assumed to be [0~1] numpy array of shape (h, w) or (1, h, w)
    '''
    if len(x.shape) >= 3:
        assert x.shape[0] == 1
        x = x[0] # from (1, h, w) to (h, w)
    x = (x * 255).astype(np.uint8)
    if eq:
        x = im_equalize(x, method='clahe')
    if alpha:
        return cv2.cvtColor(x, cv2.COLOR_GRAY2RGBA)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

def write_corners_xybs(xbs, ybs, ypredbs, output_dir, n_samples=None, eq=False):
    os.makedirs(output_dir, exist_ok=True)
    for batch, (xb, yb, ypredb) in enumerate(zip(xbs, ybs, ypredbs)):
        if n_samples:
            xb, yb, ypredb = xb[:n_samples], yb[:n_samples], ypredb[:n_samples]
        for i, (x, y, ypred) in enumerate(zip(xb, yb, ypredb)):
            im = x2rgbimg(x, eq=eq)
            y = y.reshape((-1, 2)).astype('int32')
            ypred = ypred.reshape((-1, 2)).astype('int32')
            im = draw_parallelogram(im, y, color=(0, 255, 0), thickness=1)
            im = draw_parallelogram(im, ypred, color=(255, 0, 0), thickness=1)
            if not cv2.imwrite(os.path.join(output_dir, f'b{batch}-{i}.png'), im):
                print('imwrite failed.')

def draw_parallelogram(im, pts, label=True, label_offset=5, color=255, thickness=2):
    '''
    pts: 3 points: (3, 2)
    '''
    if pts.shape[0] < 4:
        p4 = np.expand_dims(pts[0] + (pts[2] - pts[1]), axis=0) # the top right point
        pts = np.concatenate([pts, p4], axis=0)
    im = cv2.polylines(im, [pts.astype(int)], True, color, thickness)
    if label:
        for i in range(3):
            im = cv2.putText(im, str(i+1), tuple(pts[i] + label_offset),
                             cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return im

def draw_gal_centers(im, gal, color=255):
    for k in gal.header:
        if re.search('Block\d+', k) is None:
            continue
        coord = ( int(gal.header[k][0]//10), int(gal.header[k][1]//10) )
        im = cv2.circle(im, coord, 10, color=color, thickness=5)
    return im

def draw_windows(im, gal, color=255, **window_args):
    for b in range(1, gal.header['BlockCount']+1):
        p1, p2 = get_window_coords(gal, b, **window_args)
        im = cv2.rectangle(im, p1, p2, color, 2)
    return im

def draw_corners_gpr(im, gal, gpr, color=255):
    '''
    Draw all blocks in a tif
    '''
    for b in range(1, gal.header['BlockCount']+1):
        n_rows = gal.header[f'Block{b}'][Gal.N_ROWS]
        n_cols = gal.header[f'Block{b}'][Gal.N_COLS]
        pts = []
        for r, c in [[1,1], [1,n_cols], [n_rows,n_cols], [n_rows,1]]:
            x, y = gpr.data.loc[b, r, c]['X']//10, gpr.data.loc[b, r, c]['Y']//10
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        im = cv2.polylines(im, [pts], True, color=color, thickness=2)
    return im

def draw_corners_df(im, block_coords, color=255):
    for b, row in block_coords.groupby('Block'):
        pts = row.values.reshape(3, 2)
        im = draw_parallelogram(im, pts, color=color, thickness=2)
    return im

def draw_spots(im, spot_coords, color=255):
    for spot_coord in spot_coords:
        im = cv2.circle(im, (int(round(spot_coord[0])), int(round(spot_coord[1]))),
            5, color=color, thickness=2)
    return im

def draw_all_info(im_path, gal, gpr, eq=None, **window_kwargs):
    ims = read_tif(im_path, rgb=True, eq_method=eq)
    for i in range(len(ims)):
        ims[i] = draw_gal_centers(ims[i], gal, color=(255,0,0))
        ims[i] = draw_windows(ims[i], gal, color=(255,0,0), **window_kwargs)
        ims[i] = draw_corners_gpr(ims[i], gal, gpr, color=(0,255,0))
    return ims

def draw_xy_spot_coord(xb, yb):
    for i, (x, y) in enumerate(zip(xb, yb)):
        im = x2rgbimg(x)
        im[:, :, 0] = im[:, :, 0] + y*255
        im = np.clip(im, 0, 255).astype(int)
        cv2.imwrite(f'array_align/pred/test/{i}.png', im)

def draw_heatmap(im, heatmaps):
    assert im.shape[-1] == 4, print('img must be rgba')
    heatmaps = np.moveaxis(heatmaps, 0, -1).astype('float32')
    heatmaps = (cv2.cvtColor(heatmaps, cv2.COLOR_RGB2BGRA) * 255)
    im = blend_modes.dodge(im, heatmaps, .7)
    return im[:, :, :-1] # remove alpha channel

def draw_img_dir(img_dir, output_dir, gal_path, color=(255,0,0), read_tif_args={}, draw_windows_args={}):
    os.makedirs(output_dir, exist_ok=True)
    gal = Gal(gal_path)
    for f in os.listdir(img_dir):
        if not f.endswith('.tif'):
            break
        imgs = read_tif(os.path.join(img_dir, f), rgb=True, **read_tif_args)
        for channel, img in enumerate(imgs):
            img = draw_gal_centers(img, gal, color)
            img = draw_windows(img, gal, color, **draw_windows_args)
            output_path = os.path.join(output_dir, f.replace('.tif', f'_{channel}.png'))
            cv2.imwrite(output_path, img)
            print(output_path)

def draw_cohort_df_coords(cohort_df, coord_cols:List[list], colors:List[tuple]=[(255,0,0)],
        img_root='./', save_dir='./', pixel_size=10, eq=False):
    os.makedirs(save_dir, exist_ok=True)
    for path, df in cohort_df.groupby(['path']):
        ims = read_tif(os.path.join(img_root, path+'.tif'), rgb=True, eq_method=eq)
        for channel, im in enumerate(ims):
            for coord_col, color in zip(coord_cols, colors):
                im = draw_spots(im, df[coord_col].values/pixel_size, color=color)
            if eq:
                cv2.imwrite(os.path.join(save_dir, path.rsplit('/', 1)[-1] + f'_{channel}_eq.png'), im)
            else:
                cv2.imwrite(os.path.join(save_dir, path.rsplit('/', 1)[-1] + f'_{channel}.png'), im)



if __name__ == '__main__':
    pass
    # import IPython; IPython.embed(); exit()