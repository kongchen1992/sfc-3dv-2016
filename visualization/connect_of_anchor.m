function connection = connect_of_anchor(cls)

switch cls
    case 'aeroplane'
        connection = [3, 8; 2, 5; 1, 4; 6, 7];
    case 'boat'
        connection = [1, 2; 1, 3; 1, 4; 2, 3; 2, 4; 2, 5; 2, 6; 5, 6; ...
            3, 5; 4, 6; 2, 7; 3, 7; 4, 7];
    case 'bicycle'
        connection = [1, 3; 1, 7; 4, 8; 1, 11; 10, 2; 10, 6; 10, 11; ...
            5, 9; 5, 1; 9, 1; 5, 10; 9, 10];
    case 'bottle'
        connection = [1, 2; 1, 3; 1, 4; 2, 3; 2, 4; 2, 5; 3, 6; ...
            4, 7; 5, 6; 5, 7; 3, 4; 6, 7];
    case 'bus'
        connection = [1, 2; 1, 3; 3, 4; 2, 4; 1, 7; 2, 5; 3, 8; ...
            4, 6; 5, 6; 6, 8; 8, 7; 9, 11; 10, 12; 7, 5];
    case 'car'
        connection = [1, 3; 2, 4; 5, 6; 6, 7; 5, 8; 7, 8; 5, 9; ...
            6, 10; 7, 11; 8, 12; 12, 2; 2, 1; 1, 9; 11, 4; 4, 3; ...
            3, 10; 11, 12; 9, 10];
    case 'chair'
        connection = [1, 2; 1, 3; 2, 4; 3, 4; 3, 5; 4, 6; 5, 6; ...
            3, 7; 4, 8; 5, 9; 6, 10];
    case 'motorbike'
        connection = [1, 2; 2, 3; 3, 4; 5, 8; 6, 9; 7, 10; 1, 5; ...
            1, 8; 3, 6; 3, 9; 3, 5; 3, 8];
    case 'sofa'
        connection = [1, 2; 1, 5; 5, 6; 6, 2; 5, 7; 7, 9; 9, 3; 3, 1; ...
            6, 8; 8, 10; 10, 4; 4, 2; 7, 8; 9, 10; 3, 4; 5, 9; 6, 10];
    case 'diningtable'
        connection = [1, 5; 2, 6; 3, 7; 4, 8; 5, 6; 6, 8; 8, 7; 7, 5];
    case 'train'
        connection = [];% too ambiguious
        warning('Train set is not ready.')
    case 'tvmonitor'
        connection = [1, 2; 2, 4; 4, 3; 3, 1; 5, 6; 6, 8; 8, 7; ...
            7, 5; 1, 5; 2, 6; 3, 7; 4, 8];
end
