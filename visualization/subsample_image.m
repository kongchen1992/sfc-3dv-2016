function [sub_im, delta] = subsample_image(im, proj_show, pad)
im_ratio = [0.11, 0.15];
xmin = floor(min(proj_show(1, :)) - pad);
xmax = ceil(max(proj_show(1, :)) + pad);
ymin = floor(min(proj_show(2, :)) - pad);
ymax = ceil(max(proj_show(2, :)) + pad);

width = xmax - xmin;
height = ymax - ymin;
if width/im_ratio(2) < height/im_ratio(1)
    deltax = ceil((height/im_ratio(1)*im_ratio(2) - width)/2);
    xmin = xmin - deltax;
    xmax = xmax + deltax;
else
    deltay = ceil((width/im_ratio(2)*im_ratio(1) - height)/2);
    ymin = ymin - deltay;
    ymax = ymax + deltay;
end
if ymax > size(im, 1)
    deltay = ymax - size(im, 1);
    ymax = ymax - deltay;
    ymin = ymin - deltay;
end
if xmax > size(im, 2)
    deltax = xmax - size(im, 2);
    xmax = xmax - deltax;
    xmin = xmin - deltax;
end
if ymin < 1
    deltay = 1 - ymin;
    ymax = ymax + deltay;
    ymin = ymin + deltay;
end
if xmin < 1
    deltax = 1 - xmin;
    xmax = xmax + deltax;
    xmin = xmin + deltax;
end
xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(im, 2));
ymax = min(ymax, size(im, 1));
sub_im = im(ymin:ymax, xmin:xmax, :);
delta = [xmin; ymin];
