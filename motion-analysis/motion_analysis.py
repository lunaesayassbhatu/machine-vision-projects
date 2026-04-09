"""
Motion Analysis — Optical Flow & Motion Detection
Author: Luna Sbahtu | Arizona State University EEE515 Machine Vision

Implements:
  1. Dense optical flow (Farneback)
  2. Sparse optical flow (Lucas-Kanade)
  3. Background subtraction & moving object detection
  4. Motion magnitude heatmap
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ─────────────────────────────────────────────
# 1. Dense Optical Flow (Farneback)
# ─────────────────────────────────────────────
def dense_optical_flow(frame1, frame2):
    """Compute dense optical flow between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # Convert flow to HSV for visualization
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow, flow_rgb, magnitude


# ─────────────────────────────────────────────
# 2. Sparse Optical Flow (Lucas-Kanade)
# ─────────────────────────────────────────────
def sparse_optical_flow(frame1, frame2, max_corners=100):
    """Track feature points using Lucas-Kanade optical flow."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect good features to track
    corners = cv2.goodFeaturesToTrack(
        gray1, maxCorners=max_corners,
        qualityLevel=0.3, minDistance=7, blockSize=7
    )

    if corners is None:
        print("No corners detected.")
        return frame1.copy()

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    new_corners, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None, **lk_params)

    vis = frame2.copy()
    for i, (new, old) in enumerate(zip(new_corners, corners)):
        if status[i][0]:
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            cv2.arrowedLine(vis, (c, d), (a, b), (0, 255, 0), 2, tipLength=0.3)
            cv2.circle(vis, (a, b), 3, (0, 0, 255), -1)

    return vis


# ─────────────────────────────────────────────
# 3. Background Subtraction
# ─────────────────────────────────────────────
def background_subtraction(frames):
    """Detect moving objects using MOG2 background subtractor."""
    subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    masks = []
    for frame in frames:
        mask = subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masks.append(mask)
    return masks


# ─────────────────────────────────────────────
# 4. Motion Magnitude Heatmap
# ─────────────────────────────────────────────
def motion_heatmap(frames):
    """Accumulate motion magnitude across all frame pairs."""
    heatmap = np.zeros(frames[0].shape[:2], dtype=np.float32)
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        heatmap += mag
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


# ─────────────────────────────────────────────
# Demo on a static image (simulate 2 frames)
# ─────────────────────────────────────────────
def demo_on_image(image_path):
    """Run motion analysis demo by simulating motion on a single image."""
    frame1 = cv2.imread(image_path)
    if frame1 is None:
        print(f"Could not load image: {image_path}")
        return

    frame1 = cv2.resize(frame1, (512, 512))

    # Simulate motion: translate frame by a few pixels
    M = np.float32([[1, 0, 5], [0, 1, 3]])
    frame2 = cv2.warpAffine(frame1, M, (frame1.shape[1], frame1.shape[0]))

    # Dense optical flow
    flow, flow_rgb, magnitude = dense_optical_flow(frame1, frame2)

    # Sparse optical flow
    sparse_vis = sparse_optical_flow(frame1, frame2)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Dense Optical Flow (Farneback)")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(sparse_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Sparse Optical Flow (Lucas-Kanade)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("motion_analysis_results.png", dpi=150)
    plt.show()
    print("Results saved to motion_analysis_results.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = "my_image.jpg"
    demo_on_image(IMAGE_PATH)
