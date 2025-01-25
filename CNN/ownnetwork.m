% Set the path to the Digit Dataset, which is used for digit classification
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', ...
    'nndatasets', 'DigitDataset');

% Create an imageDatastore to load the images from the specified directory
% It assumes that each folder name is a label, and the images inside are examples of that label.
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, ...  % Include subfolders (for the label directories)
    'LabelSource', 'foldernames'); % Use folder names as labels for classification

% Split the dataset into training and validation sets.
% Use 750 images per class for training, and the rest for validation.
numTrainFiles = 750;
[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomized');

% Define the input size for the neural network. The images are 28x28 grayscale images.
inputSize = [28 28 1];

% Define the number of classes. There are 10 classes (digits 0 through 9).
numClasses = 10;

% Define the layers of the neural network
layers = [
    imageInputLayer(inputSize)                    % Input layer for 28x28x1 images
    convolution2dLayer(5, 20)                     % Convolutional layer with 5x5 filters and 20 filters
    batchNormalizationLayer                        % Batch normalization to improve training stability
    reluLayer                                      % ReLU activation function for non-linearity
    fullyConnectedLayer(numClasses)               % Fully connected layer to map features to class scores
    softmaxLayer                                  % Softmax layer to convert the output to probabilities
    classificationLayer                            % Classification layer for multi-class classification
];

% Set training options for the neural network
options = trainingOptions('sgdm', ...             % Stochastic gradient descent with momentum (SGDM)
    'MaxEpochs', 4, ...                           % Train for 4 epochs
    'ValidationData', imdsValidation, ...         % Use the validation data
    'ValidationFrequency', 30, ...                % Frequency of validation (every 30 iterations)
    'Verbose', false, ...                         % Disable verbose output during training
    'Plots', 'training-progress');                % Show training progress plot

% Train the neural network with the specified training data, layers, and options
net = trainNetwork(imdsTrain, layers, options);

% Classify the validation data using the trained network
YPred = classify(net, imdsValidation);

% Get the true labels for the validation data
YValidation = imdsValidation.Labels;

% Calculate the classification accuracy by comparing the predicted labels to the true labels
accuracy = mean(YPred == YValidation);

% Display the accuracy of the trained network on the validation set
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);
