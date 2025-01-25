imagesDir = 'images_256';
labelsDir = 'labels_256';

imageFiles = dir(fullfile(imagesDir, '*.jpg')); 
labelFiles = dir(fullfile(labelsDir, '*.png')); 

% get the images that have lables
imageToLabelMap = containers.Map();
for k = 1:numel(labelFiles)
    labelName = labelFiles(k).name;
    baseName = erase(labelName, '.png');
    imageToLabelMap(baseName) = fullfile(labelsDir, labelName);
end

imageNames = arrayfun(@(x) erase(x.name, '.jpg'), imageFiles, 'UniformOutput', false); 
validIndices = find(isKey(imageToLabelMap, imageNames));

filteredImgDS = subset(imageDatastore(imagesDir, 'FileExtensions', {'.jpg', '.png'}), validIndices);
filteredLabelPaths = values(imageToLabelMap, imageNames(validIndices)); 

% get classes + labels
classes = ["background", "flower"]; 
pixelLabelIDs = [3, 1]; 

filteredLabelDS = pixelLabelDatastore(filteredLabelPaths, classes, pixelLabelIDs);

if numel(filteredImgDS.Files) ~= numel(filteredLabelDS.Files)
    error('Image and label datastores must have the same number of files.');
end

combinedDS = pixelLabelImageDatastore(filteredImgDS, filteredLabelDS);

inputSize = [256, 256, 3];
numClasses = 2;

unetModel = unetLayers(inputSize, numClasses, 'ConvolutionPadding', 'same', ...
    'EncoderDepth', 4, ...
    'NumFirstEncoderFilters', 64); % U-Net architecture

options = trainingOptions('adam', ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% train U-Net
%net = trainNetwork(combinedDS, unetModel, options);
% save('segmentationExist.mat', 'net');
load('segmentationExist.mat', 'net');

% totalFiles = numel(filteredImgDS.Files);

% make a subset that includes the first half of the dataset (too much memory being used)
% halfFiles = round(totalFiles / 2);
% subsetImgDS = subset(filteredImgDS, 1:halfFiles);
% subsetLabelDS = subset(filteredLabelDS, 1:halfFiles);

% predictions + accuracy
% subsetPredictions = semanticseg(subsetImgDS, net);
% subsetAccuracy = evaluateSemanticSegmentation(subsetPredictions, subsetLabelDS);

% get ground truth and prediction to evaluate
%predictions = semanticseg(filteredImgDS, net);
%accuracy = evaluateSemanticSegmentation(predictions, filteredLabelDS);
%disp("Segmentation accuracy: " + accuracy.DataSetMetrics.GlobalAccuracy);

firstImage = readimage(filteredImgDS, 3); 
firstLabel = readimage(filteredLabelDS, 3); 

% gaussian smoothing
%gaussianSigma = 2; % Standard deviation for Gaussian kernel
%smoothedImage = imgaussfilt(firstImage, gaussianSigma);

% segmentationResult = semanticseg(firstImage, net);

% figure;

% subplot(1, 3, 1);
% imshow(firstImage);
% title('Original Image');

% subplot(1, 3, 2);
% imshow(labeloverlay(firstImage, firstLabel));
% title('Ground Truth');

% subplot(1, 3, 3);
% imshow(labeloverlay(firstImage, segmentationResult));
% title('Segmentation Result on Smoothed Image');

analyzeNetwork(net);