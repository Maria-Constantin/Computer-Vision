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

% set classes + labels
filteredLabelPaths = values(imageToLabelMap, imageNames(validIndices));
classes = ["background", "flower"];
pixelLabelIDs = [3, 1];

labelDS = pixelLabelDatastore(filteredLabelPaths, classes, pixelLabelIDs);
combinedDS = pixelLabelImageDatastore(filteredImgDS, labelDS);

layers = [
    % input
    imageInputLayer([256, 256, 3], 'Name', 'input')

    % conv 1
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_1')
    reluLayer('Name', 'relu1_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1_1')

    % conv 2
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_1')
    reluLayer('Name', 'relu2_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2_1')

    % conv 3
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_1')
    reluLayer('Name', 'relu3_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3_1') 

    % upsampling 1
    transposedConv2dLayer(2, 128, 'Stride', 2, 'Name', 'transconv1_1') 
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4_1')
    reluLayer('Name', 'relu4_1')

    % upsampling 2
    transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'transconv2_1') 
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv5_1')
    reluLayer('Name', 'relu5_1')

    % upsampling 3
    transposedConv2dLayer(2, 32, 'Stride', 2, 'Name', 'transconv3_1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv6_1')
    reluLayer('Name', 'relu6_1')

    % final conv layer
    convolution2dLayer(1, 2, 'Name', 'final_conv')
    softmaxLayer('Name', 'softmax')
    pixelClassificationLayer('Name', 'pixelClassification', 'Classes', classes, 'ClassWeights', [1, 1])
];

options = trainingOptions('rmsprop', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% train network + save
% net = trainNetwork(combinedDS, layers, options);
%save('segmentationOwn.mat', 'net');
load('segmentationOwn.mat', 'net');

% get ground truth and prediction to evaluate
%predictions = semanticseg(filteredImgDS, net);
%accuracy = evaluateSemanticSegmentation(predictions, filteredLabelDS);
%disp("Segmentation accuracy: " + accuracy.DataSetMetrics.GlobalAccuracy);

firstImage = readimage(filteredImgDS, 3); 
firstLabel = readimage(filteredLabelDS, 3); 

% Gaussian smoothing
%gaussianSigma = 2; 
%smoothedImage = imgaussfilt(firstImage, gaussianSigma);
segmentationResult = semanticseg(firstImage, net);

figure;

subplot(1, 3, 1);
imshow(firstImage);
title('Original Image');

subplot(1, 3, 2);
imshow(labeloverlay(firstImage, firstLabel));
title('Ground Truth');

subplot(1, 3, 3);
imshow(labeloverlay(firstImage, segmentationResult));
title('Segmentation Result on Smoothed Image');

analyzeNetwork(net);