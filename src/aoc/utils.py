import skimage.io
from skimage.color import rgba2rgb, rgb2gray
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_coin(image, bounding_box, resize_shape=(200, 200)):
    x1, y1, x2, y2 = bounding_box

    # Convert RGBA to RGB if needed
    if image.shape[-1] == 4:
        image = rgba2rgb(image)

    # Convert RGB to grayscale
    gray_image = rgb2gray(image)

    # Extract the coin region
    gray_image = gray_image[y1:y2, x1:x2]

    # Resize the image for easier processing (optional step)
    gray_image = skimage.transform.resize(gray_image, resize_shape)

    return gray_image


def generate_rotations(image, rotations_n):
    for i in range(rotations_n):
        degrees = i / rotations_n * 360
        rotated_image = skimage.transform.rotate(image, degrees)
        yield rotated_image, degrees


def gaussuian_mask(shape, sigma=90):
    mask = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            mask[i, j] = 1 - np.exp(
                -((i - shape[0] // 2) ** 2 + (j - shape[1] // 2) ** 2) / (2 * sigma**2)
            )
    return mask


def convolve_mask(image, mask):
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    return abs(np.fft.ifft2(fft_image * mask))


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_path, "data/20241117_213835.jpg")
    image = skimage.io.imread(image_path)

    resize_shape = (200, 200)
    bounding_box = (330, 940, 970, 1590)
    mask_sigma = 15

    image = extract_coin(image, bounding_box, resize_shape)
    mask = gaussuian_mask(resize_shape, mask_sigma)
    filtered = convolve_mask(image, mask)

    # Show the original image and the filtered image

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(filtered, cmap="gray")
    ax[1].set_title("High-pass FFT filtered")

    plt.show()
