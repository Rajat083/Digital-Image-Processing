import numpy as np

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