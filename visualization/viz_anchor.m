function viz_anchor(structure, cls, varargin)

mode = 'stick';
viewpoint = [-37.5, 30];
ground_truth = [];
color = 'b';
ivargin = 1;
while ivargin <= length(varargin)
    switch lower(varargin{ivargin})
        case 'mode'
            ivargin = ivargin + 1;
            mode = varargin{ivargin};
        case 'viewpoint'
            ivargin = ivargin + 1;
            viewpoint = varargin{ivargin};
        case 'gt'
            ivargin = ivargin + 1;
            ground_truth = varargin{ivargin};
        case 'color'
            ivargin = ivargin + 1;
            color = varargin{ivargin};
        otherwise
            fprintf('Unknown option "%s" is ignored!\n', varargin{ivargin});
    end
    ivargin = ivargin + 1;
end


switch lower(mode)
    case 'surface'
        tri = tri_of_anchor(cls);
        linewidth = 1;
        for i = 1:size(tri, 1)
            trisurf(tri, structure(1, :), structure(2, :), ...
                structure(3, :), 'LineWidth', linewidth, 'FaceColor', color, ...
                'FaceAlpha', 0.5, 'Marker', '.', 'MarkerSize', 15);
            hold on;
        end
        if ~isempty(ground_truth)
            for i = 1:size(tri, 1)
                trisurf(tri, ground_truth(1, :), ground_truth(2, :), ...
                    ground_truth(3, :), 'LineWidth', linewidth, 'FaceColor', 'r', ...
                    'FaceAlpha', 0.5, 'Marker', 'o', 'MarkerSize', 15);
                hold on;
            end
        end
    case 'stick'
        connect = connect_of_anchor(cls);
        linewidth = 3;
        for i = 1:size(connect, 1)
            plot3(structure(1, connect(i, :)), structure(2, connect(i, :)), ...
            structure(3, connect(i, :)), 'b+-', 'LineWidth', linewidth);
            hold on;
        end
    otherwise
        fprintf('Mode should be either ellipse or stick!\n');
end

axis equal vis3d off
view(viewpoint)
hold off;
