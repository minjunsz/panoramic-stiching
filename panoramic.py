import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors


def MatchSIFT(loc1: np.ndarray, des1: np.ndarray, loc2: np.ndarray, des2: np.ndarray):
    """
    Find the matches of SIFT features between two images

    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """
    neigh_des1 = NearestNeighbors(n_neighbors=2)
    neigh_des2 = NearestNeighbors(n_neighbors=2)
    neigh_des1.fit(des1)
    neigh_des2.fit(des2)
    distances, indices = neigh_des2.kneighbors(des1)
    good_idx_1, good_idx_2 = [], []
    for idx1, distance in enumerate(distances):
        if distance[0] > 0.8 * distance[1]:
            continue
        idx2 = indices[idx1][0]
        dist_opposite, idx_opposite = neigh_des1.kneighbors(des2[None, idx2])
        if dist_opposite[0][0] > 0.8 * dist_opposite[0][1]:
            continue
        if idx_opposite[0][0] == idx1:
            good_idx_1.append(idx1)
            good_idx_2.append(idx2)
    x1 = loc1[good_idx_1]
    x2 = loc2[good_idx_2]
    return x1, x2


def fillDLTRows(array: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """array is a slice of the DLT matrix, which is a 2x9 matrix"""
    array[0] = np.array([x1[0], x1[1], 1, 0, 0, 0, -x2[0] * x1[0], -x2[0] * x1[1], -x2[0]])
    array[1] = np.array([0, 0, 0, x1[0], x1[1], 1, -x2[1] * x1[0], -x2[1] * x1[1], -x2[1]])


def EstimateH(x1: np.ndarray, x2: np.ndarray, ransac_n_iter: int, ransac_thr: float):
    """
    Estimate the homography between images using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    num_matches = x1.shape[0]

    best_inlier_cnt = 0
    best_inliers: np.ndarray
    best_H: np.ndarray

    for _ in range(ransac_n_iter):
        choice = np.random.choice(num_matches, 4, replace=False)
        x1_sample = x1[choice]
        x2_sample = x2[choice]
        DLT = np.zeros((8, 9))
        for i in range(4):
            fillDLTRows(DLT[2 * i : 2 * i + 2], x1_sample[i], x2_sample[i])
        _, _, V = np.linalg.svd(DLT)
        H = V[-1].reshape(3, 3)

        homogeneous_x1 = np.concatenate([x1, np.ones((num_matches, 1))], axis=1)
        transformed = H @ homogeneous_x1.T
        transformed = transformed.T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        error = np.linalg.norm(transformed - x2, axis=1)
        inliner_idx = np.where(error < ransac_thr)[0]
        if len(inliner_idx) > best_inlier_cnt:
            best_inlier_cnt = len(inliner_idx)
            best_inliers = inliner_idx
            best_H = H

    return best_H, best_inliers


def EstimateR(H, K):
    """
    Compute the relative rotation matrix

    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """

    K_inv = np.linalg.inv(K)
    R = K_inv @ H @ K
    R = R / np.cbrt(np.linalg.det(R))
    return R


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface

    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """

    f = K[0][0]
    w = np.arange(0, Wc)
    h = np.arange(0, Hc)
    w, h = np.meshgrid(w, h, indexing="xy")
    phi = (2 * np.pi / Wc) * w
    y = h - (Hc / 2)
    x = f * np.sin(phi)
    z = f * np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane

    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """

    mask = np.ones(p.shape[0:2], dtype=bool)
    f = K[0][0]
    shape = p.shape
    p = p.reshape(-1, 3)
    transformed = (R @ p.T).T
    transformed = transformed.reshape(shape)
    mask = mask * (transformed[..., 2] > 0)

    u = transformed[..., :2] / transformed[..., 2:3] * f
    u[..., 0] += W / 2
    u[..., 1] += H / 2
    mask = mask * (u[..., 0] >= 0) * (u[..., 0] <= W) * (u[..., 1] >= 0) * (u[..., 1] <= H)

    return u, mask


def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas

    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """
    y = np.arange(0, image_i.shape[0])
    x = np.arange(0, image_i.shape[1])
    f_red = interpolate.interp2d(x, y, image_i[..., 0])
    f_green = interpolate.interp2d(x, y, image_i[..., 1])
    f_blue = interpolate.interp2d(x, y, image_i[..., 2])

    canvas_i = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    red, green, blue = [], [], []
    for h, w in u[mask_i]:
        red.append(f_red(h, w))
        green.append(f_green(h, w))
        blue.append(f_blue(h, w))
    colors = np.concatenate([red, green, blue], axis=-1)
    canvas_i[mask_i] = colors

    return canvas_i


def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image

    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """
    new_canvas = np.array(canvas)
    new_canvas[mask_i] = canvas_i[mask_i]
    return new_canvas
