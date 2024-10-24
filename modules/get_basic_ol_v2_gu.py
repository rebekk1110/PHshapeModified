import numpy as np
import gudhi as gd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def get_autooptim_bf_radius_GU(pts, down_sample_num=400, is_down=True, isDebug=False):
    if is_down:
        idx = np.random.choice(pts.shape[0], down_sample_num, replace=False)
        pts_down = pts[idx]
    else:
        pts_down = pts

    max_dist = np.max(np.linalg.norm(pts_down - np.mean(pts_down, axis=0), axis=1))

    # Calculate bfr_0d
    bfr_0d = np.linspace(0, max_dist, num=100)

    # Calculate persistent homology
    alpha_complex = gd.AlphaComplex(points=pts_down)
    simplex_tree = alpha_complex.create_simplex_tree()
    pers_0d = simplex_tree.persistence()

    if isDebug:
        print(f"[1-get_basic_ol/get_optim_bf_radius()] :: calc_PH_0d :: bfr_0d={bfr_0d}, pers_0d=\n{pers_0d}")

    pers_0d = [p for p in pers_0d if p[0] == 0]
    pers_0d = np.array([p[1] for p in pers_0d])

    pers_len_0d = pers_0d[:, 1] - pers_0d[:, 0]

    sorted_indices = np.argsort(pers_len_0d)[::-1]
    pers_0d_sorted = pers_0d[sorted_indices]
    pers_len_0d_sorted = pers_len_0d[sorted_indices]

    bfr_1d = np.linspace(0, max_dist, num=100)
    pers_1d = np.zeros_like(bfr_1d)

    for i, r in enumerate(bfr_1d):
        pers_1d[i] = np.sum(pers_0d_sorted[:, 1] > r) - np.sum(pers_0d_sorted[:, 0] > r)

    bfr_optim = bfr_1d[np.where(pers_1d == 1)[0][0]]

    if isDebug:
        print(f"[1-get_basic_ol/get_optim_bf_radius()] :: bfr_optim={bfr_optim}")

    return bfr_optim, bfr_0d, bfr_1d, pers_1d

def get_build_bf(pts, bfr_optim, bf_tole=5e-1, bf_otdiff=1e-2, isDebug=False):
    hull = ConvexHull(pts)
    bf_optim = Polygon(pts[hull.vertices])
    bf_optnew = bf_optim.buffer(-bfr_optim * bf_tole)

    if isDebug:
        print(f"[2-get_basic_ol/get_build_bf()] :: bf_optim.area={bf_optim.area}, bf_optnew.area={bf_optnew.area}")

    while (bf_optim.area - bf_optnew.area) / bf_optim.area > bf_otdiff:
        bf_optnew = bf_optnew.buffer(bfr_optim * bf_tole * 0.1)

        if isDebug:
            print(f"[2-get_basic_ol/get_build_bf()] :: bf_optim.area={bf_optim.area}, bf_optnew.area={bf_optnew.area}")

    return bf_optnew, bf_optim