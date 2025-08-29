from PIL import Image, ImageFilter
import numpy as np
import cv2

def add_gaussian_noise(image: Image.Image, level: int) -> Image.Image:
    """Adds Gaussian noise to a PIL image."""
    if level == 0:
        return image
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_cv = np.array(image)
    noise = np.random.normal(0, level, img_cv.shape).astype('uint8')
    img_cv_noisy = cv2.add(img_cv, noise)
    return Image.fromarray(img_cv_noisy)

def apply_gaussian_blur(image: Image.Image, radius: int) -> Image.Image:
    """Applies Gaussian blur to a PIL image."""
    if radius == 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))
