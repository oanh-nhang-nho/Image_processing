import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Morphological Closing (dilation followed by erosion) ===
def morphological_closing(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(img, kernel)
    closed = cv2.erode(dilated, kernel)
    return closed

# === 2. Gradient Edge Detection (Manual Sobel) ===
def gradient_edge_detection(img):
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    grad_x = cv2.filter2D(img, -1, sobel_x)
    grad_y = cv2.filter2D(img, -1, sobel_y)
    magnitude = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
    return np.uint8(np.clip(magnitude, 0, 255))

# === 3. Square Tracing Algorithm (simplified) ===
def square_tracing(binary_img):
    h, w = binary_img.shape
    contours = []
    visited = np.zeros_like(binary_img, dtype=bool)

    for y in range(1, h-1):
        for x in range(1, w-1):
            if binary_img[y, x] == 255 and not visited[y, x]:
                contour = []
                queue = [(y, x)]
                while queue:
                    cy, cx = queue.pop()
                    if visited[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    contour.append((cx, cy))
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if binary_img[ny, nx] == 255 and not visited[ny, nx]:
                                    queue.append((ny, nx))
                if len(contour) > 10:
                    contours.append(np.array(contour, dtype=np.int32))
    return contours

# === 4. SIFT-like Feature Extraction using Harris corners ===
def harris_corners(img, threshold=0.01):
    gray = np.float32(img)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    corners = np.where(dst > threshold * dst.max())
    keypoints = list(zip(corners[1], corners[0]))  # (x, y)
    return keypoints

# === 5. Simple Graph-cut-like Segmentation using k-means ===
def simple_segmentation(img, K=4):
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
    return segmented

# === MAIN FLOW ===
img = cv2.imread('R.png')  # THAY bằng ảnh của bạn
img = cv2.resize(img, (256, 256))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Closing
closed = morphological_closing(gray)

# Step 2: Gradient Edge Detection
gradient = gradient_edge_detection(closed)

# Step 3: Binary + Contour Detection
_, binary = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
contours = square_tracing(binary)
contour_img = img.copy()
for c in contours:
    for pt in c:
        cv2.circle(contour_img, tuple(pt), 1, (0, 255, 0), -1)

# Step 4: Harris-based keypoints
keypoints = harris_corners(gray)
sift_img = img.copy()
for x, y in keypoints:
    cv2.circle(sift_img, (x, y), 3, (255, 0, 0), -1)

# Step 5: K-means segmentation
segmented = simple_segmentation(img)

# === DISPLAY RESULTS ===
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.ravel()

axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Grayscale")

axes[1].imshow(closed, cmap='gray')
axes[1].set_title("Morphological Closing")

axes[2].imshow(gradient, cmap='gray')
axes[2].set_title("Gradient Edge Detection")

axes[3].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
axes[3].set_title("Contours (Square Tracing Approx)")

axes[4].imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
axes[4].set_title("Harris Corner Keypoints (SIFT-like)")

axes[5].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
axes[5].set_title("K-means Segmentation")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
