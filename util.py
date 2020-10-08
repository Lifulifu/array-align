import numpy as np
import shapely
from shapely.geometry import Polygon

def get_ious(yb, ypredb):
    ious = []
    for y, ypred in zip(yb, ypredb):
        y, ypred = to_four_pts(y), to_four_pts(ypred)
        y, ypred = Polygon(y).convex_hull, Polygon(ypred).convex_hull
        inter_area = y.intersection(ypred).area
        iou = inter_area / (y.area + ypred.area - inter_area)
        ious.append(iou)
    return np.array(ious)

def to_four_pts(pts):
    '''
    pts: 3 pts parellelgram
    returns: 4 pts (shape 4*2)
    '''
    if pts.shape != (3, 2): pts = pts.reshape(3, 2)
    fourth_pt = pts[2] + (pts[0] - pts[1])
    return np.concatenate((pts, fourth_pt.reshape(1, -1)), axis=0)

if __name__ == '__main__':
    import IPython; IPython.embed(); exit()