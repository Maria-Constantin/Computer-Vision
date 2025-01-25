close all;

%% Load training and test data using imageDatastore
% You could try adapting this to your own image sets.
syntheticDir   = fullfile(toolboxdir('vision'), 'visiondata','digits','synthetic');
handwrittenDir = fullfile(toolboxdir('vision'), 'visiondata','digits','handwritten');

trainingSet = imageDatastore(syntheticDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore("7", 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Tabulate the number of images with each label
countEachLabel(trainingSet);
countEachLabel(testSet);

%% Show some example images from the training and test sets
figure;
subplot(2, 2, 1);
imshow(trainingSet.Files{102});
title('Training Set Image 1');

subplot(2, 2, 2);
imshow(trainingSet.Files{304});
title('Training Set Image 2');

subplot(2, 2, 3); % Test Set
imshow(testSet.Files{1});
title('Test Set Image 1');

subplot(2, 2, 4);
imshow(testSet.Files{2});
title('Test Set Image 2');

%% Show pre-processing results
exTestImage1 = readimage(testSet, 1);
exTestImage2 = readimage(testSet, 2);

% Convert to grayscale and binarize the images
grayImage1 = rgb2gray(exTestImage1);
grayImage2 = rgb2gray(exTestImage2);
threshold = graythresh(grayImage1); % Otsu's method for thresholding
processedImage1 = imbinarize(grayImage1, threshold);
processedImage2 = imbinarize(grayImage2, threshold);

figure;
subplot(2, 2, 1);
imshow(exTestImage1);
title('Test Set Image 1');

subplot(2, 2, 2);
imshow(processedImage1);
title('Processed Image 1');

subplot(2, 2, 3);
imshow(exTestImage2);
title('Test Set Image 2');

subplot(2, 2, 4);
imshow(processedImage2);
title('Processed Image 2');

%% Extract HOG features and visualize (Q3: see how cell size affects the result)
img = readimage(trainingSet, 206); % Pick a random training image

% Extract HOG features with different cell sizes
[hog_2x2, vis2x2] = extractHOGFeatures(img, 'CellSize', [2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(img, 'CellSize', [4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(img, 'CellSize', [8 8]);

% Show the original image and HOG visualizations
figure;
subplot(2,3,1:3); imshow(img);

subplot(2,3,4);
plot(vis2x2);
title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});
subplot(2,3,5);
plot(vis4x4);
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});
subplot(2,3,6);
plot(vis8x8);
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

%% Set optimal cell size and extract features
cellSize = [4 4];
hogFeatureSize = length(hog_4x4);

%% Loop over the training set and extract HOG features from each image
numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    img = rgb2gray(img);
    img = imbinarize(img);  % Pre-processing step

    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

% Get labels for each image
trainingLabels = trainingSet.Labels;

%% Train classifier using SVM with one-vs-one encoding scheme
classifier = fitcecoc(trainingFeatures, trainingLabels);

%% Test the classifier
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);

% Make class predictions using the test features
predictedLabels = predict(classifier, testFeatures);

%% Tabulate the results using a confusion matrix (optional)
% confMat = confusionmat(testLabels, predictedLabels);
% helperDisplayConfusionMatrix(confMat);

%% Helper functions

function helperDisplayConfusionMatrix(confMat)
% Display the confusion matrix in a formatted table.
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));
digits = '0':'9';
colHeadings = arrayfun(@(x) sprintf('%d', x), 0:9, 'UniformOutput', false);
format = repmat('%-9s', 1, 11);
header = sprintf(format, 'digit  |', colHeadings{:});
fprintf('\n%s\n%s\n', header, repmat('-', size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s', [digits(idx) '      |']);
    fprintf('%-9.2f', confMat(idx, :));
    fprintf('\n');
end
end

function [features, setLabels] = helperExtractHOGFeaturesFromImageSet(imds, hogFeatureSize, cellSize)
% Extract HOG features from an imageDatastore.
setLabels = imds.Labels;
numImages = numel(imds.Files);
features  = zeros(numImages, hogFeatureSize, 'single');

for j = 1:numImages
    img = readimage(imds, j);
    img = imbinarize(img);  % Pre-processing step

    features(j, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end
end

