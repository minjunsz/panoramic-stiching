import cv2
import numpy as np
from matplotlib import pyplot as plt

from panoramic import (
    ConstructCylindricalCoord,
    EstimateH,
    EstimateR,
    MatchSIFT,
    Projection,
    UpdateCanvas,
    WarpImage2Canvas,
)

if __name__ == "__main__":
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([[320, 0, 480], [0, 320, 270], [0, 0, 1]])

    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = "imgs/{}.jpg".format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
        I_prev, I_current = im_list[i], im_list[i + 1]

        # Extract SIFT features
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(I_prev, None)
        kp2, des2 = sift.detectAndCompute(I_current, None)
        loc1 = np.asarray([[p.pt[0], p.pt[1]] for p in kp1])
        loc2 = np.asarray([[p.pt[0], p.pt[1]] for p in kp2])

        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)

        # Compute R_new (or R_i+1)
        R_new = R @ rot_list[i]

        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]

    Hc = Him
    Wc = len(im_list) * Wim // 2

    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure("HW1")
    plt.axis("off")
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig("output_{}.png".format(i + 1), dpi=600, bbox_inches="tight", pad_inches=0)
