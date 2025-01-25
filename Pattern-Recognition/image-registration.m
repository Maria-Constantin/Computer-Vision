% Clear workspace, close figures, and clear command window
clear; close all; clc;

%% Task 1 - Rigid Image Registration

% Read in the fixed and moving images
fixed = imread('cactus4.png');  % Read the fixed image
fixed = rgb2gray(fixed);  % Convert to grayscale

moving = imread('cactus5.png');  % Read the moving image
moving = imresize(moving, size(fixed));  % Resize moving image to match fixed image size
moving = rgb2gray(moving);  % Convert to grayscale

% Display the fixed and moving images side by side for comparison
figure;
imshowpair(fixed, moving);
title('Original Fixed and Moving Images');

% Configure the image registration optimizer and metric for rigid registration
[optimizer, metric] = imregconfig('monomodal');  % Use default config for monomodal images
optimizer.MaximumIterations = 1500;  % Set the maximum iterations for registration

% Perform rigid registration using the 'imregister' function
movingRegistered = imregister(moving, fixed, 'rigid', optimizer, metric);

% Display the fixed and registered images side by side for comparison
figure;
imshowpair(fixed, movingRegistered, 'Scaling', 'joint');
title('Rigid Registration - Fixed vs Registered Moving Image');

%% Task 2 - Affine Registration, Comparison, and Evaluation

% Perform affine registration (allows scaling, rotation, shearing) on the moving image
movingAffineRegistered = imregister(moving, fixed, 'affine', optimizer, metric);

% Display the fixed image and affine registered moving image for comparison
figure;
imshowpair(fixed, movingAffineRegistered, 'Scaling', 'joint');
title('Affine Registration - Fixed vs Affine Registered Moving Image');

% Compute the difference images between the fixed image and the registered images
diffImage = imabsdiff(fixed, movingRegistered);  % Difference after rigid registration
diffImageAffine = imabsdiff(fixed, movingAffineRegistered);  % Difference after affine registration

% Display the difference images for both rigid and affine registrations
figure;
subplot(1, 2, 1);
imshow(diffImage, []);
title('Difference after Rigid Registration');
subplot(1, 2, 2);
imshow(diffImageAffine, []);
title('Difference after Affine Registration');

% Calculate Mean Squared Error (MSE) for both rigid and affine registrations
mseRigid = immse(fixed, movingRegistered);
mseAffine = immse(fixed, movingAffineRegistered);

% Display the MSE values to compare the registration accuracies
fprintf('MSE for Rigid Registration: %.4f\n', mseRigid);
fprintf('MSE for Affine Registration: %.4f\n', mseAffine);

