# Computer Vision

## Image Processing

- This section covers various fundamental image processing techniques, including basic image manipulations (e.g., resizing, cropping), filtering (e.g., blurring, edge detection), transformations (e.g., rotation, scaling), pixel-level analysis, and segmentation for object detection and feature extraction.

## HoG Features

- Demonstrates the process of extracting **Histogram of Oriented Gradients (HoG)** features for image classification. Specifically, it showcases the use of HoG features to train a machine learning model for handwritten digit recognition.

## Pattern Recognition

- This section focuses on **image registration techniques**, highlighting **rigid** and **affine registration** methods used to align two images. Additionally, it explores **template matching techniques** such as **Sum of Squared Differences (SSD)** and **Zero-normalized Cross-correlation**, to align and track objects across images.

## Disparity Mapping

- Demonstrates the creation of a **disparity map** in stereo image matching, using **Sum of Squared Differences (SSD)** to compute the disparity between corresponding pixels from two stereo images, enabling depth perception.

## Template Matching

- Explores **template matching** for object tracking across a sequence of frames, comparing two approaches: **Fixed Template Matching** and **Template Update**. The project uses the **David Ross pgm 21mb** dataset for object tracking.

## CNN

- Implements a **Convolutional Neural Network (CNN)** to classify images of handwritten digits from the **DigitDataset**, which contains 28x28 grayscale images of digits (0-9), showcasing a deep learning-based approach for image classification.

## U-Net Segmentation Project

- This project evaluates the performance of two **U-Net-based architectures** for image segmentation tasks. It compares the standard U-Net with the Adam optimizer to a modified U-Net using the RMSprop optimizer. The project focuses on segmenting flowers from backgrounds using the **Oxford Flower Dataset**, concluding that the RMSprop-based U-Net outperforms the Adam-based one in terms of accuracy, training time, and noise reduction.