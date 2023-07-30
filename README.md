#  Digital-Image-Processing-Filters
Welcome to ImagePro-Denoise! This repository showcases algorithms and code implementations dedicated to enhancing image quality by reducing noise and artifacts. Whether you're interested in exploring state-of-the-art denoising methods or contributing your own, ImagePro-Denoise is the place to collaborate and discover the world of image processing and restoration.
By the way the image used inn this notebook is a picture of memorial tomb of the great scientist Avicenna , located in IRAN,HAMEDAN.

## Table of Contents
- [Introduction](#introduction)
- [Implemented Noises](#implemented-noises)
- [Implemented Filters](#implemented-filters)
- [Point Processing Methods](#point-processing-methods)
- [Histogram Equalization](#histogram-equalization)
- [Comparing Original and Denoised Images](#comparing-original-and-denoised-images)
- [Contributing](#contributing)


## Introduction
This repository focuses on digital image processing with a specific emphasis on  techniques for noising and denoising. The goal is to develop and showcase algorithms that can significantly improve image quality by reducing various types of noise and artifacts. Whether you're a beginner or an expert in image processing, ImagePro-Denoise offers valuable insights and code implementations to enhance your understanding and skills.

## Implemented Noises
In  Digital-Image-Processing-Filters, I introduce several types of noise that can degrade image quality. These include:

1. Gaussian Noise: Additive white noise that affects pixel values.
2. Salt and Pepper Noise: Impulse noise that randomly sets pixels to either maximum or minimum intensity.
3. Speckle Noise: Multiplicative noise that affects pixel values multiplicatively.
4. Poisson Noise: Noise that follows a Poisson distribution, often present in low-light images.

For each noise type, I provide  visual examples, and the corresponding implementation in code.

## Implemented Filters
To tackle the challenge of denoising, ImagePro-Denoise presents various filtering techniques, including:

1. Spatial Filters
2. Frequency Filters
   
Each filter is thoroughly explained, and code examples demonstrate how to apply them to denoise images.

## Point Processing Methods
Point processing methods are simple image processing techniques that treat each pixel independently.Examples include contrast stretching, histogram equalization, gamma correction, and thresholding.

## Histogram Equalization
Histogram equalization is an image processing technique used to enhance image contrast and reduce noise by redistributing the pixel intensities across the full dynamic range. The method works by transforming the pixel values in such a way that the resulting image has a more uniform histogram.

## Comparing Original and Denoised Images
at the end, I emphasize the importance of quantifying denoising effectiveness. I utilize the Structural Similarity Index (SSIM) to compare the denoised images with their original counterparts. This objective evaluation helps to assess the performance of different denoising methods and algorithms.

SSIM Score histo:', 0.88
![SSIM Score](https://github.com/ShimaKosarifard/Digital-Image-Processing-Filters/blob/main/Images/RESULT.png)


## Contributing
I welcome contributions from the community to make ImagePro-Denoise even better! If you have implemented new denoising algorithms, improved existing ones, or have ideas for enhancing the repository, please feel free to submit pull requests. For major changes, please open an issue first to discuss the proposed modifications.
Here is an example: 





