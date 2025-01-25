%% Image Processing
% This section demonstrates basic image processing, including reading an image,
% converting it to grayscale, cropping, and displaying histograms of the images.

% Reading the image 'wheat.png'
i = imread("wheat.png");

% Convert the image to grayscale
i_grey = rgb2gray(i);

% Optional: Crop the image, but the line is commented out here
% i_cropped = imcrop(i_grey);

% Display histograms of the original image, grayscale image, and cropped image
subplot(1, 3, 1), imhist(i), title('Original Image');
subplot(1, 3, 2), imhist(i_grey), title('Grayscale Image');
subplot(1, 3, 3), imhist(i_cropped), title('Cropped Image');

%% Image Filtering and Transformation
% This section applies image filtering and transformations to the grayscale image.

% Apply a Gaussian filter with a standard deviation of 3
i_gaussian = imgaussfilt(i_grey, 3);

% Apply a median filter with a 5x5 neighborhood
i_median = medfilt2(i_grey, [5 5]);

% Optional: Display the original image, Gaussian-filtered image, and median-filtered image
% subplot(1, 3, 1), imshow(i_grey), title('Original Image');
% subplot(1, 3, 2), imshow(i_gaussian), title('Gaussian Filtered Image');
% subplot(1, 3, 3), imshow(i_median), title('Median Filtered Image');

% Rotate the grayscale image by 90 degrees
i_rotate = imrotate(i_grey, 90);

% Optional: Display the rotated image (commented out)
% imshow(i_rotate);

% Flip the image horizontally (mirror effect)
i_mirror = flip(i, 2);

% Display the mirrored image
imshow(i_mirror);

%% Image Analysis and Saving
% This section demonstrates pixel-level analysis and saving the processed image.

% Displaying pixel information using imtool
imtool(i);

% Optional: Save the mirrored image as 'mirrored_wheat.png'
% imwrite(i_mirror, "mirrored_wheat.png");

%% Image Segmentation
% This section demonstrates image segmentation by applying smoothing, thresholding,
% and quantization techniques.

% Reading the image 'eight.tif'
i5 = imread("eight.tif");

% Apply a Gaussian filter to smooth the image and remove noise
i5_gaussian = imgaussfilt(i5, 3);

% Apply a median filter to the Gaussian-filtered image
i5_median = medfilt2(i5_gaussian, [5 5]);

% Threshold the image to segment it into different regions
i5_thresh = multithresh(i5_median);

% Quantize the image based on the thresholds
i5_seg = imquantize(i5_median, i5_thresh);

% Convert the labeled segmented image to an RGB image for visualization
i5_segRGB = label2rgb(i5_seg);

% Display the segmented image in RGB format
figure;
imshow(i5_segRGB);
axis off;
title('RGB Segmented Image');

% Save the segmented RGB image as 'coins_seg.png'
imwrite(i5_segRGB, "coins_seg.png");
