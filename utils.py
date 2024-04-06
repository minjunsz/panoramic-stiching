import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_correspondence(img1: np.ndarray, img2: np.ndarray, kp1: np.ndarray, kp2: np.ndarray):
    """Draw lines between matched key points.

    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.
    kp1 : np.ndarray
        Matched key points in the first image [k,2]
    kp2 : np.ndarray
        Matched key points in the second image [k,2]
    """
    full_img = cv2.hconcat([img1, img2])
    shifted_kp2 = kp2 + np.array([img1.shape[1], 0])
    int_kp1 = np.array(kp1, dtype=np.int32)
    int_kp2 = np.array(shifted_kp2, dtype=np.int32)
    for idx in range(kp1.shape[0]):
        full_img = cv2.line(full_img, int_kp1[idx], int_kp2[idx], color=(0, 255, 0), thickness=1)
    plt.imshow(full_img)
    plt.show()
