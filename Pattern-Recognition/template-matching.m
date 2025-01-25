clear; close all; clc;

% Read in images
fixed = imread('cactus3.png');
fixed = rgb2gray(fixed); % Convert to grayscale if not already

moving  = imread('cactus3_crop1.png');
moving = rgb2gray(moving);

% Resize the fixed image to match the size of the moving image (optional but recommended)
fixed = imresize(fixed, [size(moving, 1), size(moving, 2)]);

% Set template size (the size of the moving image)
sz = 100;

%% SSD Template Matching (Euclidean distance based)
img = double(fixed)./255;  % Normalize fixed image
moving = double(moving)./255;  % Normalize moving image

% Initialize score matrix to store error values
score = zeros(size(img, 1)-sz, size(img, 2)-sz);

% Loop through the fixed image to compute SSD (Sum of Squared Differences)
for ii = 1:size(img, 1)-sz
    for jj = 1:size(img, 2)-sz
        tar = img(ii:ii+sz-1, jj:jj+sz-1);  % Extract the target subimage
        score(ii, jj) = sum((moving(:) - tar(:)).^2); % Squared difference
    end
end

% Display the SSD error map
figure;
imagesc(score);
colorbar;
title('SSD Template Matching Error Map');

% Find the position of the minimum error (best match)
[posx, posy] = find(score == min(min(score)));
best_match_ED = [posx posy];

% Show the best match rectangle on the error map
hold on;
rectangle('Position', [posy, posx, sz, sz], 'LineWidth', 2, 'EdgeColor', 'r');
hold off;

% Show the best match rectangle on the original image
figure;
imshow(fixed);
hold on;
rectangle('Position', [posy, posx, sz, sz], 'LineWidth', 2, 'EdgeColor', 'r');
hold off;

%% Zero-normalized Cross-correlation Template Matching
img = double(fixed)./255;  % Normalize fixed image
moving = double(moving)./255;  % Normalize moving image

% Initialize score matrix for cross-correlation
score = zeros(size(img, 1)-sz, size(img, 2)-sz);

% Normalize the template (moving image) by removing the mean and normalizing
tmplt1 = moving(:) - mean(moving(:));
tmplt1 = tmplt1 ./ norm(tmplt1);

% Loop through the fixed image to compute zero-normalized cross-correlation
for ii = 1:size(img, 1)-sz
    for jj = 1:size(img, 2)-sz
        tar = img(ii:ii+sz-1, jj:jj+sz-1);  % Extract the target subimage
        tar = tar(:) - mean(tar(:));  % Zero-mean normalization for the target
        tar = tar ./ norm(tar);  % Normalize the target
        score(ii, jj) = tmplt1(:)' * tar(:);  % Cross-correlation calculation
    end
end

% Display the cross-correlation score map
figure;
imagesc(score);
colorbar;
title('Zero-normalized Cross-correlation Matching');

% Find the position of the maximum score (best match)
[posx, posy] = find(score == max(max(score)));
best_match_Corr = [posx posy];

% Show the best match rectangle on the cross-correlation map
hold on;
rectangle('Position', [posy, posx, sz, sz], 'LineWidth', 2, 'EdgeColor', 'r');
hold off;

% Show the best match rectangle on the original image
figure;
imshow(fixed);
hold on;
rectangle('Position', [posy, posx, sz, sz], 'LineWidth', 2, 'EdgeColor', 'r');
hold off;
