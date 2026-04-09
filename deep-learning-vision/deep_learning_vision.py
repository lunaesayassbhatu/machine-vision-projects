"""
Deep Learning Vision — Image Classification & Object Detection
Author: Luna Sbahtu | Arizona State University EEE515 Machine Vision

Uses HuggingFace Transformers for:
  1. Zero-shot image classification
  2. Object detection with bounding boxes
  3. Image segmentation
  4. Depth estimation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from transformers import pipeline

# ─────────────────────────────────────────────
# Load image
# ─────────────────────────────────────────────
def load_image(path, size=(512, 512)):
    image = Image.open(path).convert("RGB")
    image = image.resize(size)
    return image


def show_image(image, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 1. Image Classification
# ─────────────────────────────────────────────
def classify_image(image):
    print("\n=== Image Classification ===")
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    results = classifier(image)
    for r in results[:5]:
        print(f"  {r['label']:40s} {r['score']:.2%}")

    labels = [r['label'].split(',')[0] for r in results[:5]]
    scores = [r['score'] for r in results[:5]]

    plt.figure(figsize=(8, 4))
    plt.barh(labels[::-1], scores[::-1], color='steelblue')
    plt.xlabel("Confidence")
    plt.title("Top-5 Classification Results")
    plt.tight_layout()
    plt.savefig("classification_results.png", dpi=150)
    plt.show()
    return results


# ─────────────────────────────────────────────
# 2. Object Detection
# ─────────────────────────────────────────────
def detect_objects(image):
    print("\n=== Object Detection ===")
    detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    results = detector(image)

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    colors = plt.cm.get_cmap('tab10', len(results))
    for i, r in enumerate(results):
        box = r['box']
        x, y = box['xmin'], box['ymin']
        w = box['xmax'] - box['xmin']
        h = box['ymax'] - box['ymin']

        rect = patches.Rectangle((x, y), w, h,
                                  linewidth=2,
                                  edgecolor=colors(i),
                                  facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"{r['label']} {r['score']:.0%}",
                color=colors(i), fontsize=9, fontweight='bold')
        print(f"  {r['label']:20s} confidence={r['score']:.2%}  box={box}")

    ax.set_title("Object Detection")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("object_detection.png", dpi=150)
    plt.show()
    return results


# ─────────────────────────────────────────────
# 3. Depth Estimation
# ─────────────────────────────────────────────
def estimate_depth(image):
    print("\n=== Depth Estimation ===")
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
    result = depth_estimator(image)
    depth_map = np.array(result["depth"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(depth_map, cmap="plasma")
    axes[1].set_title("Depth Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Depth")

    plt.tight_layout()
    plt.savefig("depth_estimation.png", dpi=150)
    plt.show()
    return depth_map


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = "your_image.jpg"

    image = load_image(IMAGE_PATH)
    show_image(image, "Input Image")

    classify_image(image)
    detect_objects(image)
    estimate_depth(image)

    print("\nAll results saved as PNG files.")
