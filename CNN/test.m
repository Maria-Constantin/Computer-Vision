net = alexnet;

layer = 6;
name = net.Layers(layer).Name;

channels = 1:36;
I = deepDreamImage(net, name, channels, 'PyramidLevels',1);

% I = imread("peppers.png");
% inputSize = net.Layers(1).InputSize;
% I = imresize(I, inputSize(1:2));

% label = classify(net,I);
% figure
% imshow(I)
% title(string(label))
% deepNetworkDesigner(net)

figure
I = imtile(I, 'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer', name, 'Features'], 'Interpreter','none')
analyzeNetwork(net)