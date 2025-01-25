clear; close all; clc
basepath = 'frames/';

%% Read the image files
filenames = dir([basepath '*.pgm']);
filenames = sort({filenames.name});

im = imread([basepath filenames{1}]);
data = repmat(uint8(0), [size(im, 1), size(im, 2), length(filenames)]);
for ii = 1:length(filenames)
    data(:,:,ii) = imread([basepath filenames{ii}]);
end

%% Plot Frame 1 and draw rectangle
figure('Name', "Frame 1");
imagesc(squeeze(data(:,:,1)));
colormap(gray)

hold on
rectangle('Position',[142, 76, 30, 30], 'LineWidth', 2, 'EdgeColor', 'r')
hold off

%% Extract Template from Frame 1
start_r = 76;
start_c = 142;
sz = 32;
tmplt = data(start_r:start_r+sz-1, start_c:start_c+sz-1, 1);
figure('Name', "Template");
imagesc(tmplt);
colormap(gray)

%% Version 1: Fixed Template Matching
% In this version, the template is not updated. It remains fixed throughout the frames.

% Initialize template (normalized)
img = double(data(:,:,1)) / 255;
tmplt = double(tmplt) / 255;
score = zeros(size(img, 1) - sz, size(img, 2) - sz);
tmplt1 = tmplt(:) - mean(tmplt(:));
tmplt1 = tmplt1 / norm(tmplt1);

% Loop through frames
close all
figure;
tracking_fig = subplot(1, 1, 1);
colormap(gray)

for dd = 1:length(filenames)
    img = double(data(:,:,dd));

    for ii = 1:size(img, 1) - sz
        for jj = 1:size(img, 2) - sz
            tar = img(ii:ii+sz-1, jj:jj+sz-1);
            tar = tar(:) - mean(tar(:));
            tar = tar / norm(tar);
            score(ii, jj) = tmplt1(:)' * tar(:);
        end
    end

    [posx, posy] = find(score == max(max(score)));

    % Display Tracking Result
    hold on
    imagesc(img, 'Parent', tracking_fig);
    rectangle('Position', [posy, posx, 30, 30], 'LineWidth', 2, 'EdgeColor', 'r', 'Parent', tracking_fig)
    hold off
    pause(0.5)
end


%% Version 2: Template Update
% In this version, the template is updated each time based on the position of the best match.

% Initialize template (normalized)
img = double(data(:,:,1)) / 255;
tmplt = double(tmplt) / 255;
score = zeros(size(img, 1) - sz, size(img, 2) - sz);
tmplt1 = tmplt(:) - mean(tmplt(:));
tmplt1 = tmplt1 / norm(tmplt1);

close all
figure;
tracking_fig = subplot(1, 1, 1);
colormap(gray)

figure;
template_fig = subplot(1, 1, 1);
colormap(gray)

% Loop through frames and update template
for dd = 1:length(filenames)
    img = double(data(:,:,dd));

    for ii = 1:size(img, 1) - sz
        for jj = 1:size(img, 2) - sz
            tar = img(ii:ii+sz-1, jj:jj+sz-1);
            tar = tar(:) - mean(tar(:));
            tar = tar / norm(tar);
            score(ii, jj) = tmplt1(:)' * tar(:);
        end
    end

    [posx, posy] = find(score == max(max(score)));

    % Display Tracking Result
    hold on
    imagesc(img, 'Parent', tracking_fig);
    rectangle('Position', [posy, posx, 30, 30], 'LineWidth', 2, 'EdgeColor', 'r', 'Parent', tracking_fig)
    hold off
    pause(0.5)

    % Update Template
    start_r = posx;
    start_c = posy;
    tmplt1 = data(start_r:start_r+sz-1, start_c:start_c+sz-1, dd);

    % Display Updated Template
    imagesc(tmplt1, 'Parent', template_fig);

    tmplt1 = double(tmplt1);
    tmplt1 = tmplt1(:) - mean(tmplt1(:));
    tmplt1 = tmplt1 / norm(tmplt1);

    drawnow
end
