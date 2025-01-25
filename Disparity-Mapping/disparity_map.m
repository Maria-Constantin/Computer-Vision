clear all;
close all;

%% Read the images
imgLeft = imread("left.png");
imgRight = imread("right.png");

%% Setting up the hyperparams
disparityMax = 64;  % Max disparity value
windowSize = 21;    % Search window size
h = floor(windowSize/2);  % Half of the window size

%% Convert the images to grayscale
leftSide = rgb2gray(imgLeft);
rightSide = rgb2gray(imgRight);

%% Some sanity checks
if ~all(size(leftSide) == size(rightSide))
    error("Images have different sizes");
end

%% Initialize disparity map
[height, width] = size(leftSide);
disparityMap = single(zeros([height, width]));

%% Loop through the image and compute the disparity map using SSD
for y = 1 + h : height - h
    for x = 1 + h : width - h
        % Get the patch from the left image
        leftPatch = leftSide(y - h : y + h, x - h : x + h);

        % Initialize variables to store the best match
        bestDifference = Inf;
        bestDisparity = 0;

        % Loop through the search window in the right image
        for d = 0 : disparityMax
            % Define the search range in the right image
            xRight = x - d;
            if xRight < 1 + h || xRight > width - h
                % Skip if the search goes beyond image boundaries
                continue;
            end

            % Get the patch from the right image
            rightPatch = rightSide(y - h : y + h, xRight - h : xRight + h);

            % Calculate the sum of squared differences (SSD)
            difference = sum((leftPatch(:) - rightPatch(:)).^2); % SSD

            % Update the best match if the current disparity has a lower difference
            if difference < bestDifference
                bestDifference = difference;
                bestDisparity = d;
            end
        end

        % Store the best disparity in the disparity map
        disparityMap(y, x) = bestDisparity;
    end
end

%% Post-processing (optional) to reduce noise
% Apply a median filter to smooth the disparity map and reduce noise
disparityMap = medfilt2(disparityMap, [5, 5]);

%% Display the results
% Normalize the disparity map for better visualization
disparityMap_display = disparityMap / max(disparityMap(:));

% Display the disparity map with a jet colormap
figure;
imshow(disparityMap_display, []);
colormap jet;
colorbar;
title('Disparity Map using SSD');
