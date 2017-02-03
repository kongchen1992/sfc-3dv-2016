#Structure from Category: A Generic and Prior-less Approach
This repository contains necessary code for the structure from category method (SfC) presented at 3DV 2016.

##Prerequisites
##### Torch
We use Torch 7 (http://torch.ch) for our implementation with these additional packages:
- [`mattorch`](https://github.com/clementfarabet/lua---mattorch): we use `.mat` file for saving input 2D projection matrix and estimated 3D structure.
- [`cutorch`](https://github.com/torch/cutorch): we utilize GPU to accelerate matrix manipulation.
- [`MAGMA`](http://icl.cs.utk.edu/magma/): we utilize CUDA implementation of `gels`.

####Visualization
MATLAB (tested on R2016b)


##Installation
Our current release has been tested on Ubuntu 16.04 LTS.

####Cloning the repository
```sh
git clone https://github.com/kongchen1992/sfc-3dv-2016.git
```

####Download PASCAL3D+ (optional)
A series of annotated 2D chairs is included in this repository, which is stored in `data/chair_pascal.mat`.
As a result, PASCAL3D+ dataset is not required in terms of running the demo.
However, if you want see the images where these 2D annotations come from and compare the estimated 3D structures against them, you have to download [`PASCAL3D+`](ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip) and pass the path to the function `visualize.m`.

## Guide
####Running demo
The implementation of Alternating Direction Method of Multipliers (ADMMs) is included in the file `structure_from_category.lua`.

To run the demo, open torch and simply run
```sh
th> dofile('demo.lua')
```
The result should be saved under the folder `data/results/`.

####Visualization
We used MATLAB to visualize the 3D skeleton.
After obtaining the estimated 3D structure, simply add the folder `visualization` to MATLAB's path and run `visualize.m` for 3D plot.

The function `visualize.m` takes one required input, the object category (e.g. 'chair'), and two options.

Options include
- `frames`: a subset/order of frames to visualize;
- `pascal_dir`: the path to Pascal3D+ dataset. If this option is empty, the 2D images will be prohibited to show.

For details and examples, run in MATLAB:
```
>> help visualize
```

## Reference

    @inproceedings{kong2016structure,
      title={Structure from Category: A Generic and Prior-less Approach},
      author={Kong, Chen and Zhu, Rui and Kiani, Hamed and Lucey, Simon},
      booktitle={3D Vision (3DV), 2016 Fourth International Conference on},
      pages={296--304},
      year={2016},
      organization={IEEE}
    }

For any questions, please contact Chen Kong (chenk@cs.cmu.edu).
