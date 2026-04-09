"""
Image Processing — FFT, Edge Detection, Frequency Filtering & Image Blending
Author: Luna Sbahtu | Arizona State University EEE515 Machine Vision

Implements:
  1. Image downsampling and upsampling (nearest neighbor & bilinear)
  2. FFT frequency analysis (magnitude & phase)
  3. Low-pass and high-pass frequency filtering
  4. Edge detection — Sobel, Laplacian, custom Canny
  5. Gaussian and Laplacian pyramid construction
  6. Multi-resolution image blending
  7. Hybrid image generation
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs("figs", exist_ok=True)


def show(img, title, cmap=None, save=True):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(f"figs/{title.lower().replace(' ', '_')}.png", dpi=120)
    plt.show()


# ─────────────────────────────────────────────
# 1. Sampling
# ─────────────────────────────────────────────
def downsample(img, factor=2):
    return img[::factor, ::factor]

def upsample_nearest(img, factor=2):
    return img.repeat(factor, axis=0).repeat(factor, axis=1)

def upsample_bilinear(img, factor=2):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────
# 2. FFT Analysis
# ─────────────────────────────────────────────
def fft_analysis(gray):
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    magnitude = np.log1p(np.abs(Fshift))
    phase = np.angle(Fshift)
    return Fshift, magnitude, phase

def apply_mask(Fshift, mask):
    filtered = Fshift * mask
    return np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))

def circular_mask(shape, radius, low_pass=True):
    h, w = shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    return (dist <= radius).astype(float) if low_pass else (dist > radius).astype(float)


# ─────────────────────────────────────────────
# 3. Edge Detection
# ─────────────────────────────────────────────
def edge_detection(gray):
    # Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = (sobel / sobel.max() * 255).astype(np.uint8)

    # Laplacian
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8)

    # Canny
    canny = cv2.Canny(gray, 50, 150)

    return sobel, laplacian, canny


# ─────────────────────────────────────────────
# 4. Gaussian & Laplacian Pyramids
# ─────────────────────────────────────────────
def gaussian_pyramid(img, levels=4):
    pyr = [img.copy()]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyr.append(img)
    return pyr

def laplacian_pyramid(gauss_pyr):
    pyr = []
    for i in range(len(gauss_pyr) - 1):
        size = (gauss_pyr[i].shape[1], gauss_pyr[i].shape[0])
        expanded = cv2.pyrUp(gauss_pyr[i + 1], dstsize=size)
        lap = cv2.subtract(gauss_pyr[i], expanded)
        pyr.append(lap)
    pyr.append(gauss_pyr[-1])
    return pyr


# ─────────────────────────────────────────────
# 5. Multi-resolution Blending
# ─────────────────────────────────────────────
def multiresolution_blend(img1, img2, levels=4):
    gp1 = gaussian_pyramid(img1.astype(np.float32), levels)
    gp2 = gaussian_pyramid(img2.astype(np.float32), levels)
    lp1 = laplacian_pyramid(gp1)
    lp2 = laplacian_pyramid(gp2)

    blended = []
    for l1, l2 in zip(lp1, lp2):
        h, w = l1.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:, :w//2] = 1.0
        if l1.ndim == 3:
            mask = mask[:, :, np.newaxis]
        blended.append(l1 * mask + l2 * (1 - mask))

    result = blended[-1]
    for i in range(len(blended) - 2, -1, -1):
        size = (blended[i].shape[1], blended[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + blended[i]

    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 6. Hybrid Image
# ─────────────────────────────────────────────
def hybrid_image(img1, img2, sigma_low=5, sigma_high=10):
    low = cv2.GaussianBlur(img1.astype(np.float32), (0, 0), sigma_low)
    blurred = cv2.GaussianBlur(img2.astype(np.float32), (0, 0), sigma_high)
    high = img2.astype(np.float32) - blurred
    hybrid = np.clip(low + high, 0, 255).astype(np.uint8)
    return hybrid


# ─────────────────────────────────────────────
# Main Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Load a sample image (replace with your own)
    img = cv2.imread("Outdoorimage.png")
    if img is None:
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        print("No image found — using random data for demo")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Sampling
    down = downsample(gray)
    up_nn = upsample_nearest(down)
    up_bi = upsample_bilinear(down)
    show(gray, "Original Grayscale", cmap="gray")
    show(down, "Downsampled", cmap="gray")
    show(up_nn, "Upsampled Nearest", cmap="gray")
    show(up_bi, "Upsampled Bilinear", cmap="gray")

    # 2. FFT
    Fshift, mag, phase = fft_analysis(gray)
    show(mag, "FFT Magnitude", cmap="gray")
    show(phase, "FFT Phase", cmap="gray")

    low_mask = circular_mask(gray.shape, radius=30, low_pass=True)
    high_mask = circular_mask(gray.shape, radius=30, low_pass=False)
    show(apply_mask(Fshift, low_mask), "Low-Pass Filtered", cmap="gray")
    show(apply_mask(Fshift, high_mask), "High-Pass Filtered", cmap="gray")

    # 3. Edge Detection
    sobel, laplacian, canny = edge_detection(gray)
    show(sobel, "Sobel Edges", cmap="gray")
    show(laplacian, "Laplacian Edges", cmap="gray")
    show(canny, "Canny Edges", cmap="gray")

    # 4. Pyramids
    gp = gaussian_pyramid(gray)
    lp = laplacian_pyramid(gp)
    show(gp[1], "Gaussian Pyramid Level 1", cmap="gray")
    show(lp[0], "Laplacian Pyramid Level 0", cmap="gray")

    # 5. Blending
    h, w = rgb.shape[:2]
    img1 = rgb.copy()
    img2 = np.zeros_like(rgb)
    img2[:, :, 0] = 255  # red image for demo
    blended = multiresolution_blend(img1, img2)
    show(blended, "Multi-resolution Blend")

    print("All figures saved to figs/")
