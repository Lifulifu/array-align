
from tqdm import tqdm
from skimage.transform import radon
import numpy as np
import matplotlib.pyplot as plt
import cv2

from gridding.util import *
from gridding.dataset import *

def get_local_gridding_imgs(img_path, gal: Gal, gpr: Gpr, pixel_size=10):
    ims = read_tif(img_path)
    x_padding = gal.header['Block1'][Gal.COL_MARGIN] // pixel_size
    y_padding = gal.header['Block1'][Gal.ROW_MARGIN] // pixel_size
    idxs, result, window_coords = [], [], []
    for b in range(1, gal.header['BlockCount']+1):
        idxs.append([img_path, b])
        coord_df = gpr.data.loc[b][['X', 'Y']] // pixel_size
        p1 = [int(coord_df['X'].values.min() - x_padding),
              int(coord_df['Y'].values.min() - y_padding)]
        p2 = [int(coord_df['X'].values.max() + x_padding),
              int(coord_df['Y'].values.max() + y_padding)]
        im = np.stack([crop_window(im, p1, p2) for im in ims])
        result.append(im)
        window_coords.append((p1, p2))
    return idxs, result, window_coords


def median_filter_3D(im, kernel_size):
    '''
    img: (c, h, w)
    '''
    h, w = im.shape[1:]
    result_h = h - kernel_size[0] + 1
    result_w = w - kernel_size[1] + 1
    result = np.zeros((result_h, result_w))
    for i in range(result_h):
        for j in range(result_w):
            result[i, j] = np.median(
                im[:, i:i+kernel_size[0], j:j+kernel_size[1]])
    return result  # (h, w)


def mask_radon(im, step=1):
    '''
    mask out values outside the inscribed circle
    to satisfy assumptions of radon()
    '''
    image = np.copy(im)
    shape_min = min(image.shape)
    radius = shape_min // 2
    img_shape = np.array(image.shape)
    coords = np.array(
        np.ogrid[:image.shape[0], :image.shape[1]], dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    image[outside_reconstruction_circle] = 0
    return radon(image, theta=np.arange(0, 180, step))


def get_entropy(a):
    return -np.sum(a * np.log(a))


def get_tilt_angle(im, max_tilt=30, resolution=10):
    '''
    max_tilt: in degrees
    resolution: bins per degree
    '''
    trans = mask_radon(im, 1/resolution)
    trans /= trans.sum(axis=0)  # normalize
    trans += 1e-128  # to prevent divide by 0
    entropies = np.zeros(trans.shape[1])
    for angle in range(trans.shape[1]):
        entropies[angle] = get_entropy(trans[:, angle])
    entropies[np.isnan(entropies)] = np.inf

    max_tilt = int(max_tilt * resolution)
    left_min = np.argmin(entropies[:max_tilt])
    right_min = len(entropies) - max_tilt + np.argmin(entropies[-max_tilt:])
    if entropies[left_min] < entropies[right_min]:
        angle = left_min
    else:
        angle = right_min - 1800
    return angle/resolution, trans


def tilt_correct(im, max_tilt=30, resolution=10):
    angle, _ = get_tilt_angle(im, max_tilt, resolution)
    return im_rotate(im, -angle)


def im_sharpen(im, kernal_size=3, rate=.1):
    '''
    im: [0, 255]
    high pass = im - blurred_im
    result = im + high pass * rate
    '''
    hp = np.clip(im - cv2.GaussianBlur(im,
                                       (kernal_size, kernal_size), 0), 0, 255)
    return np.clip(hp * rate + im, 0, 255)


def get_peaks(hist):
    '''
    run otsu and get centers of the line segments
    '''
    hist = (hist / hist.max() * 255).astype('uint8')
    thres, binarized = cv2.threshold(
        hist, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binarized = binarized.flatten()

    peaks = []
    start = 0
    for i in range(1, len(binarized)):
        if binarized[i-1] != binarized[i]:
            if binarized[i-1] == 0:  # 0 -> 1, rising edge
                start = i
            else:  # 1 -> 0, dropping edge
                peaks.append((start + i) // 2)
    return peaks


def refine_peaks(peaks, sp_tolerance=.2):
    '''
    rough implementation of Joseph's alg
    '''
    spacings = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
    med_sp = np.median(spacings)
    min_sp = med_sp - med_sp * sp_tolerance
    max_sp = med_sp + med_sp * sp_tolerance

    less_count = 0
    result = [peaks[0]]
    for i in range(len(spacings)):
        if (spacings[i] >= min_sp) and (spacings[i] < max_sp):
            if less_count == 1:
                result.append(peaks[i])
                less_count = 0
            result.append(peaks[i+1])
        elif spacings[i] < min_sp:
            less_count += 1
        elif spacings[i] >= max_sp:
            n_skipped = int(spacings[i] / med_sp)
            for j in range(n_skipped):
                result.append(peaks[i] + med_sp * (j+1))
            result.append(peaks[i+1])
        if less_count >= 2:
            result.append(peaks[i+1])
            less_count = 0
    return np.array(result).astype(int)


def draw_gridlines(im, x_lines, y_lines, x_color=(0, 0, 255), y_color=(0, 255, 0), thickness=1):
    im = x2rgbimg(im)
    for x in x_lines:
        im = cv2.polylines(im, [
            np.array([[x, 0], [x, im.shape[0]]])], True, x_color, thickness)
    for y in y_lines:
        im = cv2.polylines(im, [
            np.array([[0, y], [im.shape[1], y]])], True, y_color, thickness)
    return im


def rotate_coords(coords, angle, origin):
    angle = np.deg2rad(angle)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(coords)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def joseph_predict_pipeline(img_path, gal: Gal, gpr: Gpr,
                            pixel_size=10, kernel_size=(3, 3), max_tilt=30):

    idxs, imgs, window_coords = get_local_gridding_imgs(img_path, gal, gpr, pixel_size)
    result_idxs, spots = [], []
    for idx, img, window_coord in tqdm(zip(idxs, imgs, window_coords), total=len(idxs)):
        # median filtering
        img = median_filter_3D(img, kernel_size)
        # img = img.max(axis=0).astype(np.uint8)

        # top hat filtering
        # kernel = np.ones((3, 3),np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

        # morph opening
        # kernel = np.ones((5, 5),np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # eq
        img = im_equalize(img, 'clahe')

        # tilt correction
        angle, _ = get_tilt_angle(img, max_tilt)
        img = im_rotate(img, -angle)

        # refine projections
        x_hist, y_hist = img.mean(axis=0), img.mean(axis=1)
        x_peaks = get_peaks(x_hist)
        y_peaks = get_peaks(y_hist)
        x_refined = refine_peaks(x_peaks)
        y_refined = refine_peaks(y_peaks)

        img = draw_gridlines(img, x_peaks, y_peaks)
        cv2.imwrite(f'gridding/imgs/test/rep_test/{idx[1]}.png', img)

        im_path, b = idx
        n_cols = gal.header[f'Block{b}'][Gal.N_COLS]
        n_rows = gal.header[f'Block{b}'][Gal.N_ROWS]
        block_spots = []
        for c in range(1, n_cols+1):
            for r in range(1, n_rows+1):
                result_idxs.append((im_path, b, c, r))
                try:
                    block_spots.append([x_refined[c-1], y_refined[r-1]])
                except:
                    block_spots.append([0, 0])
        p1, p2 = window_coord
        win_h, win_w = (p2[0] - p1[0]), (p2[1] - p1[1])
        block_spots = rotate_coords(np.array(block_spots), -angle, origin=(win_h/2, win_w/2))
        block_spots += np.array(p1)
        spots.append(block_spots)

    result_idxs = pd.MultiIndex.from_tuples(result_idxs).set_names(
        ['img_path', 'Block', 'Column', 'Row'])
    spots = np.concatenate(spots)
    return pd.DataFrame(spots, index=result_idxs, columns=['x', 'y'])
