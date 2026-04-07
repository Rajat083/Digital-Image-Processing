import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(img):
    hist = np.zeros(256, dtype=int)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            intensity = img[i, j]
            hist[intensity] += 1
    return hist/(rows*cols)


def histogram_equalization(img):
    pdf = compute_histogram(img)
    cdf = np.zeros(256)
    cdf[0] = pdf[0]

    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]

    mapping = np.round(cdf * 255).astype(np.uint8)

    equalized = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equalized[i, j] = mapping[img[i, j]]

    return equalized, mapping


def apply_filter(image, kernel):
    kh, kw = kernel.shape
    h, w, c = image.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
    
    res_img = np.zeros_like(image, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded_img[i : i + kh, j : j + kw, :]
            res_img[i, j, :] = np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1))
            
    return res_img

def print_image(image):
    plt.imshow(image, cmap='gray')
    plt.title(f'Image Shape: {image.shape}')
    plt.axis('off')
    plt.show()