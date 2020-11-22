import cv2
import numpy as np

from .preprocess import *

def x2rgbimg(x):
    x = x[0] # from (c, h, w) to (h, w, c)
    x = (x * 255).astype(np.float32)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

def draw_parallelogram(im, pts, label=True, label_offset=5, color=255):
    '''
    pts: 3 points: (x1, y1, x2, y2, x3, y3)
    '''
    pts = pts.reshape(3, 2)
    p4 = np.expand_dims(pts[0] + (pts[2] - pts[1]), axis=0) # the top right point
    pts = np.concatenate([pts, p4], axis=0)
    im = cv2.polylines(im, [pts], True, color, 2)
    if label:
        for i in range(3):
            im = cv2.putText(im, str(i+1), tuple(pts[i] + label_offset),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
    return im

def draw_gal_centers(im, gal, color=255):
    for k in gal.header:
        if re.search('Block\d+', k) is None:
            continue
        coord = ( gal.header[k][0]//10, gal.header[k][1]//10 )
        im = cv2.circle(im, coord, 10, color=color, thickness=5)
    return im

def draw_windows(im, gal, color=255, **window_args):
    for b in range(1, gal.header['BlockCount']+1):
        p1, p2 = get_window_coords(b, gal, **window_args)
        im = cv2.rectangle(im, p1, p2, color, 2)
    return im

def draw_gt_blocks(im, gal, gpr, color=255):
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

def draw_all_info(im_path, gal_path, gpr_path, eq=None, eq_kwargs=None):
    gal, gpr = Gal(gal_path), Gpr(gpr_path)
    ims = read_tif(im_path, rgb=True, eq=eq, eq_kwargs=eq_kwargs)
    for i in range(len(ims)):
        ims[i] = draw_gal_centers(ims[i], gal, color=(255,0,0))
        ims[i] = draw_windows(ims[i], gal, color=(255,0,0))
        ims[i] = draw_gt_blocks(ims[i], gal, gpr,  color=(0,255,0))
    return ims

def draw_xy(xb, yb):
    count = 0
    for x, y in zip(xb.detach().numpy(), yb.detach().numpy()):
        im = x2rgbimg(x)
        im = draw_parallelogram(im, y.astype(int), color=(0,255,0))
        cv2.imwrite(f'garbage/{count}.png', im)
        count += 1

def draw_spots(im, spot_df, color=255):
    for idx, row in spot_df.iterrows():
        im = cv2.circle(im, (int(round(row['x'])), int(round(row['y']))),
            5, color=color, thickness=2)
    return im

if __name__ == '__main__':
    pass
    # import IPython; IPython.embed(); exit()