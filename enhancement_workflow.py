"""
GenAI-Powered Medical Image Enhancement
---------------------------------------
This script improves medical ultrasound images by:
  • Removing noise
  • Enhancing sharpness and clarity
  • Computing PSNR and SSIM metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, color, transform, img_as_float, filters, restoration, util, metrics

# Step 1: Load and prepare medical ultrasound images
def load_ultrasound_images(base_dir="./Dataset_BUSI_with_GT", max_images=3):
    categories = ["benign", "malignant", "normal"]
    images, names = [], []

    for cat in categories:
        folder = os.path.join(base_dir, cat)
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(folder, fname)
                img = io.imread(path)
                if img.ndim == 3:
                    img = color.rgb2gray(img)
                img = transform.resize(img_as_float(img), (256, 256), anti_aliasing=True)
                images.append(img)
                names.append(f"{cat}_{fname}")
                if len(images) >= max_images:
                    break
        if len(images) >= max_images:
            break

    if not images:
        print("No medical images found! Please check your dataset path.")
    else:
        print(f"Loaded {len(images)} ultrasound images successfully.")
    return images, names

# Step 2: Simulate degradation (blur + Gaussian noise)
def degrade_image(image):
    blurred = filters.gaussian(image, sigma=2.0)
    noisy = util.random_noise(blurred, mode="gaussian", var=0.01)
    return np.clip(noisy, 0, 1)

# Step 3: Enhance images using denoising + deconvolution
def enhance_image(noisy_img):
    sigma = np.mean(restoration.estimate_sigma(noisy_img, channel_axis=None))
    denoised = restoration.denoise_nl_means(
        noisy_img,
        h=1.15 * sigma,
        patch_size=5,
        patch_distance=6,
        fast_mode=True,
        channel_axis=None
    )

    # Create a simple PSF (Gaussian)
    size = 9
    x = np.linspace(-4, 4, size)
    xv, yv = np.meshgrid(x, x)
    psf = np.exp(-(xv**2 + yv**2) / (2 * (1.0**2)))
    psf /= psf.sum()

    try:
        deconv, _ = restoration.unsupervised_wiener(denoised, psf)
        deconv = np.clip(deconv, 0, 1)
    except Exception:
        deconv = denoised

    return denoised, deconv

# Step 4: Compute metrics and visualize results
def compare_and_display(name, original, noisy, denoised, enhanced, save_dir):
    result = {
        "Image": name,
        "PSNR_Noisy": round(metrics.peak_signal_noise_ratio(original, noisy, data_range=1.0), 3),
        "SSIM_Noisy": round(metrics.structural_similarity(original, noisy, data_range=1.0), 4),
        "PSNR_Denoised": round(metrics.peak_signal_noise_ratio(original, denoised, data_range=1.0), 3),
        "SSIM_Denoised": round(metrics.structural_similarity(original, denoised, data_range=1.0), 4),
        "PSNR_Enhanced": round(metrics.peak_signal_noise_ratio(original, enhanced, data_range=1.0), 3),
        "SSIM_Enhanced": round(metrics.structural_similarity(original, enhanced, data_range=1.0), 4)
    }

    # Visualization (4 images side by side)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["Original", "Degraded", "Denoised", "Enhanced"]
    imgs = [original, noisy, denoised, enhanced]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(name)
    plt.tight_layout()
    plt.show()

    # Save output
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{name}_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return result


# Step 5: Run workflow
def run_enhancement_workflow():
    save_dir = "./enhanced_outputs"
    images, names = load_ultrasound_images()

    if not images:
        return

    all_results = []

    for name, img in zip(names, images):
        noisy = degrade_image(img)
        denoised, enhanced = enhance_image(noisy)
        metrics_dict = compare_and_display(name, img, noisy, denoised, enhanced, save_dir)
        all_results.append(metrics_dict)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    print("\nEnhancement metrics saved successfully!\n")
    print(df)

# Step 6: Execute
if __name__ == "__main__":
    run_enhancement_workflow()
    print("\nWorkflow completed: Medical images enhanced and displayed.")
