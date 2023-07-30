import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import requests
from io import BytesIO
import sys
from PIL import Image



def show_img_cv(img_title, img):
    cv2.imshow(img_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_img_plt(img, c_map ='gray', dpi=100, fig_hight=8, fig_width=6, vmin=0, vmax=255):
    plt.figure(figsize=(fig_hight, fig_width), dpi=dpi)
    plt.imshow(img, cmap=c_map, vmin=vmin, vmax=vmax)

def display_hist_plt(img, bins=256, range=(0, 256)):
    plt.figure(figsize=(4, 2), dpi=100)
    plt.hist(img.flat, bins=bins, range=range)
    plt.show()
  
def show_mult_img(rows, columns, img_names, img_titles, vmin=0, vmax=255):
    fig = plt.figure(figsize=(15, 17), dpi=100)
    for i in range(len(img_names)):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img_names[i], cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(img_titles[i])

class ImageNoise:
    """
    Class to add different types of noise to an image. The types of noise that can be added are:
    - Gaussian
    - Gaussian for RGB images
    - Salt and Pepper
    - Poisson
    - Speckle

    The image is passed in when creating an object of the class.

    Parameters
    ----------
    image : ndarray
        The input image.
    """
    def __init__(self, image):
        self.image = image

    def add_gauss_noise(self, **kwargs):
        """
        Add Gaussian noise to the image.

        Parameters
        ----------
        mean : float, optional
            Mean of the Gaussian distribution to generate noise (default is 0).
        var : float, optional
            Variance of the Gaussian distribution to generate noise (default is 0.1).
        """
        mean = kwargs.get('mean', 0)
        var = kwargs.get('var', 0.1)
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, self.image.shape)
        return self.image + self.image * gauss

    def add_sp_noise(self, **kwargs):
        """
        Add Salt & Pepper noise to the image.

        Parameters
        ----------
        s_vs_p : float, optional
            Ratio of salt to pepper (default is 0.5).
        amount : float, optional
            Overall proportion of image pixels to replace with noise (default is 0.004).
        """
        s_vs_p = kwargs.get('s_vs_p', 0.5)
        amount = kwargs.get('amount', 0.004)
        out = np.copy(self.image)

        # Salt mode
        num_salt = np.ceil(amount * self.image.size * s_vs_p).astype(int)
        coords = tuple(map(lambda dim: np.random.randint(0, dim, num_salt), self.image.shape))
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * self.image.size * (1. - s_vs_p)).astype(int)
        coords = tuple(map(lambda dim: np.random.randint(0, dim, num_pepper), self.image.shape))
        out[coords] = 0

        return out

    def add_poisson_noise(self, **kwargs):
        """
        Add Poisson noise to the image.

        The noise is added as per a Poisson distribution. This function does not take any additional parameters.
        """
        # Convert the image to double data type
        image = self.image.astype(np.float64)

        # Scale the image to the range of 0-1
        image /= np.max(image)

        # Convert the image to represent counts in the range of 0-255
        image *= 255

        # Apply the Poisson noise
        noisy = np.random.poisson(image)

        # Normalize the noisy image
        noisy = noisy / np.max(noisy)

        noisy *= 255.
        noisy = noisy.astype(np.uint8)

        return noisy

    def add_speckle_noise(self, **kwargs):
        """
        Add Speckle noise to the image.

        Speckle noise is a multiplicative noise. This function does not take any additional parameters.
        """
        gauss = np.random.randn(*self.image.shape)
        return self.image + self.image * gauss

    def add_noise(self, noise_typ, **kwargs):
        """
        Add noise to the image.

        Parameters
        ----------
        noise_typ : str
            Type of noise to add. Options are 'gauss', 's&p', 'poisson', or 'speckle'.
        **kwargs :
            Additional parameters for the noise functions. These depend on the type of noise.
        """
        match noise_typ:
            case "gauss":
                return self.add_gauss_noise(**kwargs)
            case "s&p":
                return self.add_sp_noise(**kwargs)
            case "poisson":
                return self.add_poisson_noise(**kwargs)
            case "speckle":
                return self.add_speckle_noise(**kwargs)
            case _:
                raise ValueError(f"Noise type '{noise_typ}' is not supported")

def add_gaussianRGB(image, mean=0, std=50):
    noise = np.random.normal(mean, std, size=image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

    b, g, r = cv2.split(image)

    b_noisy = add_gaussianRGB(b)
    g_noisy = add_gaussianRGB(g)
    r_noisy = add_gaussianRGB(r)

    noisy_image = cv2.merge([b_noisy, g_noisy, r_noisy])

class SpatialFilter :
    def __init__(self, image):
        self.image = image

    def mean_filter(self, filter_size, padding=0):
        padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        return cv2.blur(padded_image, (filter_size, filter_size))

    def median_filter(self, filter_size, padding=0):
        padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        return cv2.medianBlur(padded_image, filter_size)

    def adaptive_filter(self, filter_size, padding=0):
        padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        return cv2.adaptiveThreshold(padded_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, filter_size, 2)

    def apply_filter(self, filter_name, filter_size, padding=0):

        match filter_name:
            case 'mean':
                return self.mean_filter(filter_size, padding)
            case 'median':
                return self.median_filter(filter_size, padding)
            case 'adaptive':
                return self.adaptive_filter(filter_size, padding)
            case _:
                raise ValueError(f"Filter '{filter_name}' is not supported")
                
class FrequencyFilterRGB:
    def __init__(self, image):
        self.image = image

    def fftshift(self):
        np_image = np.array(self.image)

        # Separate the RGB channels
        red_channel = np_image[:, :, 0]
        green_channel = np_image[:, :, 1]
        blue_channel = np_image[:, :, 2]

        # Apply FFT to each channel
        red_fft = np.fft.fft2(red_channel)
        green_fft = np.fft.fft2(green_channel)
        blue_fft = np.fft.fft2(blue_channel)

        # Apply FFT shift to each channel
        red_fft_shifted = np.fft.fftshift(red_fft)
        green_fft_shifted = np.fft.fftshift(green_fft)
        blue_fft_shifted = np.fft.fftshift(blue_fft)

        # Combine the shifted channels back into a single image
        fftshifted_image = np.stack((red_fft_shifted, green_fft_shifted, blue_fft_shifted), axis=-1)

        return fftshifted_image

    def ifftshift(self, fshift):
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        return img_back

    def create_mask(self, radius):
        base = np.zeros(self.image.shape[:2], dtype=np.uint8)
        center_x = self.image.shape[1] // 2
        center_y = self.image.shape[0] // 2
        cv2.circle(base, (center_x, center_y), int(radius), 1, -1)
        return base

    def low_pass_filter(self, radius):
        # Apply FFT shift on the RGB image
        fshifted_image = self.fftshift()

        # Create the low-pass filter mask
        mask = self.create_mask(radius)

        # Separate the shifted channels
        red_channel_shifted = fshifted_image[:, :, 0]
        green_channel_shifted = fshifted_image[:, :, 1]
        blue_channel_shifted = fshifted_image[:, :, 2]

        # Apply the mask to each shifted channel in the frequency domain
        red_channel_filtered_shifted = red_channel_shifted * mask
        green_channel_filtered_shifted = green_channel_shifted * mask
        blue_channel_filtered_shifted = blue_channel_shifted * mask

        # Apply IFFT shift to each filtered channel
        red_channel_filtered_restored = self.ifftshift(red_channel_filtered_shifted)
        green_channel_filtered_restored = self.ifftshift(green_channel_filtered_shifted)
        blue_channel_filtered_restored = self.ifftshift(blue_channel_filtered_shifted)

        # Combine the restored channels back into a single image
        filtered_image = np.stack((red_channel_filtered_restored, green_channel_filtered_restored, blue_channel_filtered_restored), axis=-1)

        # Convert the numpy array back to a PIL image for visualization or further processing
        filtered_image_pil = Image.fromarray(np.uint8(filtered_image))

        return filtered_image_pil

    def high_pass_filter(self, radius):
        # Apply FFT shift on the RGB image
        fshifted_image = self.fftshift()

        # Create the high-pass filter mask (complement of the low-pass filter mask)
        mask = self.create_mask(radius)
        high_pass_mask = 1 - mask

        # Separate the shifted channels
        red_channel_shifted = fshifted_image[:, :, 0]
        green_channel_shifted = fshifted_image[:, :, 1]
        blue_channel_shifted = fshifted_image[:, :, 2]

        # Apply the high-pass filter mask to each shifted channel in the frequency domain
        red_channel_filtered_shifted = red_channel_shifted * high_pass_mask
        green_channel_filtered_shifted = green_channel_shifted * high_pass_mask
        blue_channel_filtered_shifted = blue_channel_shifted * high_pass_mask

        # Apply IFFT shift to each filtered channel
        red_channel_filtered_restored = self.ifftshift(red_channel_filtered_shifted)
        green_channel_filtered_restored = self.ifftshift(green_channel_filtered_shifted)
        blue_channel_filtered_restored = self.ifftshift(blue_channel_filtered_shifted)

        # Combine the restored channels back into a single image
        filtered_image = np.stack((red_channel_filtered_restored, green_channel_filtered_restored, blue_channel_filtered_restored), axis=-1)

        # Convert the numpy array back to a PIL image for visualization or further processing
        filtered_image_pil = Image.fromarray(np.uint8(filtered_image))

        return filtered_image_pil

    def band_pass_filter(self, min_radius, max_radius):
        # Apply FFT shift on the RGB image
        fshifted_image = self.fftshift()

        # Create masks for the band-pass filter
        min_mask = self.create_mask(min_radius)
        max_mask = self.create_mask(max_radius)
        band_mask = min_mask - max_mask

        # Separate the shifted channels
        red_channel_shifted = fshifted_image[:, :, 0]
        green_channel_shifted = fshifted_image[:, :, 1]
        blue_channel_shifted = fshifted_image[:, :, 2]

        # Apply the band-pass filter mask to each shifted channel in the frequency domain
        red_channel_filtered_shifted = red_channel_shifted * band_mask
        green_channel_filtered_shifted = green_channel_shifted * band_mask
        blue_channel_filtered_shifted = blue_channel_shifted * band_mask

        # Apply IFFT shift to each filtered channel
        red_channel_filtered_restored = self.ifftshift(red_channel_filtered_shifted)
        green_channel_filtered_restored = self.ifftshift(green_channel_filtered_shifted)
        blue_channel_filtered_restored = self.ifftshift(blue_channel_filtered_shifted)

        # Combine the restored channels back into a single image
        filtered_image = np.stack((red_channel_filtered_restored, green_channel_filtered_restored, blue_channel_filtered_restored), axis=-1)

        # Convert the numpy array back to a PIL image for visualization or further processing
        filtered_image_pil = Image.fromarray(np.uint8(filtered_image))

        return filtered_image_pil

    def band_reject_filter(self, min_radius, max_radius):
        # Apply FFT shift on the RGB image
        fshifted_image = self.fftshift()

        # Create masks for the band-reject filter
        min_mask = self.create_mask(min_radius)
        max_mask = self.create_mask(max_radius)
        band_mask = min_mask - max_mask

        # Separate the shifted channels
        red_channel_shifted = fshifted_image[:, :, 0]
        green_channel_shifted = fshifted_image[:, :, 1]
        blue_channel_shifted = fshifted_image[:, :, 2]

        # Apply the band-reject filter mask to each shifted channel in the frequency domain
        red_channel_filtered_shifted = red_channel_shifted * (1 - band_mask)
        green_channel_filtered_shifted = green_channel_shifted * (1 - band_mask)
        blue_channel_filtered_shifted = blue_channel_shifted * (1 - band_mask)

        # Apply IFFT shift to each filtered channel
        red_channel_filtered_restored = self.ifftshift(red_channel_filtered_shifted)
        green_channel_filtered_restored = self.ifftshift(green_channel_filtered_shifted)
        blue_channel_filtered_restored = self.ifftshift(blue_channel_filtered_shifted)

        # Combine the restored channels back into a single image
        filtered_image = np.stack((red_channel_filtered_restored, green_channel_filtered_restored, blue_channel_filtered_restored), axis=-1)

        # Convert the numpy array back to a PIL image for visualization or further processing
        filtered_image_pil = Image.fromarray(np.uint8(filtered_image))

        return filtered_image_pil

def fftshift(image):
    f = np.fft.fft2(image, axes=(0, 1))  # Compute the 2D FFT along the first two axes (height and width)
    fshift = np.fft.fftshift(f, axes=(0, 1))  # Perform the FFT shift along the first two axes
    return fshift

def ifftshift(fshift):
    fishift = np.fft.ifftshift(fshift, axes=(0, 1))  # Perform the inverse FFT shift along the first two axes
    img_back = np.fft.ifft2(fishift, axes=(0, 1))  # Compute the 2D inverse FFT along the first two axes
    img_back = np.abs(img_back)  # Get the magnitude of the inverse FFT result
    return img_back

class PointProcessing:
    def __init__(self, image):
        self.image = image

    def apply_negative(self):
        """
        Applies negative transformation to the image.
        :return: Negative image.
        """
        if len(self.image.shape) == 2:
            negative_image = 255 - self.image
        else:
            # For RGB images, we need to apply negative to each channel separately
            negative_image = np.zeros_like(self.image)
            for channel in range(self.image.shape[-1]):
                negative_image[:, :, channel] = 255 - self.image[:, :, channel]

        return negative_image

    def apply_threshold(self, threshold_value):
        """
        Applies a binary threshold to the image.
        :param threshold_value: Threshold value for binarization.
        :return: Binary image.
        """
        if len(self.image.shape) == 2:
            _, binary_image = cv2.threshold(self.image, threshold_value, 255, cv2.THRESH_BINARY)
        else:
            # Convert the RGB image to grayscale for thresholding
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        
        return binary_image
    def apply_gamma_correction(self, gamma):
        """
        Applies gamma correction to the image.
        :param gamma: Gamma value for correction.
        :return: Gamma-corrected image.
        """
        if len(self.image.shape) == 2:
            gamma_corrected = np.array(255 * (self.image / 255) ** gamma, dtype='uint8')
        else:
            # For RGB images, we need to apply gamma correction to each channel separately
            gamma_corrected = np.zeros_like(self.image)
            for channel in range(self.image.shape[-1]):
                gamma_corrected[:, :, channel] = np.array(255 * (self.image[:, :, channel] / 255) ** gamma, dtype='uint8')

        return gamma_corrected

    def apply_log_transform(self, c=1):
        """
        Applies logarithmic transformation to the image.
        :param c: Constant for scaling the transformation.
        :return: Log-transformed image.
        """
        if len(self.image.shape) == 2:
            log_transformed = c * np.log(1 + self.image)
        else:
            # For RGB images, we need to apply log transform to each channel separately
            log_transformed = np.zeros_like(self.image)
            for channel in range(self.image.shape[-1]):
                log_transformed[:, :, channel] = c * np.log(1 + self.image[:, :, channel])

        return np.array(log_transformed, dtype='uint8')

    def apply_contrast_stretching(self, r_min=0, r_max=255):
        """
        Applies contrast stretching to the image.
        :param r_min: Minimum intensity value after stretching.
        :param r_max: Maximum intensity value after stretching.
        :return: Stretched image.
        """
        if len(self.image.shape) == 2:
            stretched = np.interp(self.image, (np.min(self.image), np.max(self.image)), (r_min, r_max))
        else:
            # For RGB images, we need to apply contrast stretching to each channel separately
            stretched = np.zeros_like(self.image)
            for channel in range(self.image.shape[-1]):
                stretched[:, :, channel] = np.interp(self.image[:, :, channel], (np.min(self.image[:, :, channel]), np.max(self.image[:, :, channel])), (r_min, r_max))

        return np.array(stretched, dtype='uint8')


class HistogramEnhancement:
    def __init__(self, image):
        self.image = image

    def apply_histogram_equalization(self):
        """
        Applies histogram equalization to the image.
        :return: Histogram equalized image.
        """
        if len(self.image.shape) == 2:
            histogram_equalized = cv2.equalizeHist(self.image)
        else:
            # For RGB images, we need to apply histogram equalization to each channel separately
            histogram_equalized = np.zeros_like(self.image)
            for channel in range(self.image.shape[-1]):
                histogram_equalized[:, :, channel] = cv2.equalizeHist(self.image[:, :, channel])

        return histogram_equalized

    def apply_histogram_matching(self, reference_image):
        """
        Applies histogram matching to the image using a reference image.
        :param reference_image: Reference image for histogram matching.
        :return: Histogram matched image.
        """
        # Apply histogram equalization to both the source and reference images
        equalized_source_image = self.apply_histogram_equalization()
        equalized_reference_image = HistogramEnhancement(reference_image).apply_histogram_equalization()

        # Convert the images to the HSV color space
        source_hsv = cv2.cvtColor(equalized_source_image, cv2.COLOR_BGR2HSV)
        reference_hsv = cv2.cvtColor(equalized_reference_image, cv2.COLOR_BGR2HSV)

        # Calculate the histograms of the Value channel for both images
        source_hist, _ = np.histogram(source_hsv[:, :, 2], bins=256, range=[0, 256])
        reference_hist, _ = np.histogram(reference_hsv[:, :, 2], bins=256, range=[0, 256])

        # Calculate cumulative distribution functions (CDFs)
        source_cdf = source_hist.cumsum()
        reference_cdf = reference_hist.cumsum()
        source_cdf_normalized = source_cdf / float(source_cdf.max())
        reference_cdf_normalized = reference_cdf / float(reference_cdf.max())

        # Create a lookup table to map pixel values
        lookup_table = np.interp(source_cdf_normalized, reference_cdf_normalized, np.arange(0, 256))

        # Apply the lookup table to the Value channel of the source image
        matched_value_channel = np.interp(source_hsv[:, :, 2], np.arange(0, 256), lookup_table).astype(np.uint8)

        # Merge the matched Value channel with the original Hue and Saturation channels
        matched_hsv = cv2.merge([source_hsv[:, :, 0], source_hsv[:, :, 1], matched_value_channel])

        # Convert the matched image back to BGR color space
        matched_bgr = cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2BGR)

        return matched_bgr


   