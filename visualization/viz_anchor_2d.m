function viz_anchor(projection, cls, varargin)

ground_truth = [];
mode = 'point';
ivargin = 1;
gt_points = [];
while ivargin <= length(varargin)
    switch lower(varargin{ivargin})
        case 'gt'
            ivargin = ivargin + 1;
            ground_truth = varargin{ivargin};
        case 'gt_point'
            ivargin = ivargin + 1;
            gt_points = varargin{ivargin};
        case 'mode'
            ivargin = ivargin + 1;
            mode = varargin{ivargin};
        otherwise
            fprintf('Unknown option "%s" is ignored!\n', varargin{ivargin});
    end
    ivargin = ivargin + 1;
end

switch mode
case 'point'
    linewidth = 3;
    hold on;
    plot(projection(1, :), projection(2, :), 'g.', 'MarkerSize', 60);
    if ~isempty(ground_truth)
        plot(ground_truth(1, :), ground_truth(2, :), 'ro', 'MarkerSize', 15);
        for i = 1:size(projection, 2)
            line([ground_truth(1, i); projection(1, i)], ...
                [ground_truth(2, i); projection(2, i)])
        end
    end
case 'stick'
    connect = connect_of_anchor(cls);
    linewidth = 3;
    for i = 1:size(connect, 1)
        plot(projection(1, connect(i, :)), projection(2, connect(i, :)), ...
        'b+-', 'LineWidth', linewidth);
        hold on;
    end
case 'surface'
    tri = tri_of_anchor(cls);
    linewidth = 1;
    for i = 1:size(tri, 1)
        fill(projection(1, tri(i, :)), projection(2, tri(i, :)), ...
            'b', 'LineWidth', linewidth, ...
            'FaceAlpha', 0.5, 'Marker', '.', 'MarkerSize', 15);
        hold on;
    end
    if ~isempty(ground_truth)
        for i = 1:size(tri, 1)
            fill(ground_truth(1, tri(i, :)), ground_truth(2, tri(i, :)), ...
                'r', 'LineWidth', linewidth, ...
                'FaceAlpha', 0.5, 'Marker', '.', 'MarkerSize', 15);
            hold on;
        end
    end
    if ~isempty(gt_points)
        plot(projection(1, :), projection(2, :), 'g.', 'MarkerSize', 40);
        plot(gt_points(1, :), gt_points(2, :), 'ro', 'MarkerSize', 20, ...
            'linewidth', 3);
        hold off;
    end
end

axis equal off
hold off;
