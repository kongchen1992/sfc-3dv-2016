function visualize(cls, varargin)
%VISUALIZE 3D structure visualization.
%
%    VISUALIZE(CLASS) visualize estimated 3D structures in all frames of 
%    object category CLASS.
%
%    VISUALIZE(CLASS, NAME, VALUE) specifies properties, including:
%        'frames'        a subset or certain order of frames;
%        'pascal_dir'    the path to pascal3D+ dataset to show images.
%    
%    Example
%        visualize('chair');
%        visualize('chair', 'frames', [1,4,6]);
%        visualize('chair', 'pascal_dir', 'your/path/to/PASCAL3D/');

ip = inputParser;
addOptional(ip, 'frames', []);
addOptional(ip, 'pascal_dir', []);
parse(ip, varargin{:});
frames = ip.Results.frames;
pascal_dir = ip.Results.pascal_dir;

mat_content = load(fullfile('data', 'results', [cls, '.mat']));
struc_est = mat_content.S_est';
proj_est = mat_content.W_est';

% load ground truth
info_file = fullfile('data', [cls, '_pascal.mat']);
mat_content = load(info_file);
W_gt = mat_content.W;
Name = mat_content.Name;
Gamma = mat_content.Gamma;
struc_gt = mat_content.S;

% normalize structure for visualization
num_frames = size(struc_est, 1)/3;
struc_gt = bsxfun(@minus, struc_gt, mean(struc_gt, 2));
struc_est = bsxfun(@minus, struc_est, mean(struc_est, 2));
scale = sum(reshape(sum(struc_est.*struc_gt, 2), 3, num_frames), 1) ./ ...
    sum(reshape(sum(struc_est.*struc_est, 2), 3, num_frames), 1);
struc_est_rot = kron(diag(scale), eye(3))*struc_est;

if isempty(frames)
    frames = 1:num_frames;
end

for iframe = 1:numel(frames)
    i = frames(iframe);
    fprintf('Frame %d...\n', i)

    if ~isempty(pascal_dir)
        images_path = fullfile(pascal_dir, 'Images');
        image_dir = fullfile(images_path, [cls, '_pascal'], [Name{i},'.jpg']);
        im = imread(image_dir);
        figure(1);
        proj_show = proj_est(2*i-1:2*i, :);
        proj_show(2, :) = -proj_show(2, :);
        proj_gt = W_gt(2*i-1:2*i, :);
        proj_gt(2, :) = -proj_gt(2, :);
        [sub_im, delta] = subsample_image(im, proj_show, 10);
        proj_show = proj_show -  repmat(delta, 1, size(proj_show, 2));
        proj_gt = proj_gt -  repmat(delta, 1, size(proj_gt, 2));
        hold off;
        imshow(sub_im);
        hold on;
        proj_gt(find(proj_gt < 0)) = nan;
        viz_anchor_2d(proj_show, cls, 'mode', 'surface', 'gt_point', proj_gt);
    else
        warning(['No path to Pascal3D+ dataset specified.',  ...
            'Image showing prohibited.'])
    end

    figure(2);
    viz_anchor(struc_est_rot(3*i-2:3*i, :), cls, 'mode', 'surface', ...
        'viewpoint', [0, 90]);

    pause;
end
