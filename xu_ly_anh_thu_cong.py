
import numpy as np
import cv2

def lbp_pixel(img, x, y):
    center = img[x, y]
    code = 0
    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]:
        neighbor = img[x+dx, y+dy]
        code = (code << 1) | (1 if neighbor >= center else 0)
    return code

def compute_lbp(img):
    h, w = img.shape
    lbp_img = np.zeros((h-2, w-2), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            lbp_img[i-1, j-1] = lbp_pixel(img, i, j)
    return lbp_img



import numpy as np
import cv2

def convolve(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    result = np.zeros((h-2, w-2))
    for i in range(1, h-1):
        for j in range(1, w-1):
            region = img[i-1:i+2, j-1:j+2]
            result[i-1, j-1] = np.sum(region * kernel)
    return result

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])



import numpy as np
import cv2

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kmeans_segmentation(image, k=3, max_iter=10):
    pixels = image.reshape(-1, 3).astype(float)
    centers = pixels[np.random.choice(len(pixels), k, replace=False)]

    for _ in range(max_iter):
        labels = np.array([np.argmin([euclidean(p, c) for c in centers]) for p in pixels])
        for i in range(k):
            if np.any(labels == i):
                centers[i] = np.mean(pixels[labels == i], axis=0)

    segmented = centers[labels].reshape(image.shape).astype(np.uint8)
    return segmented
