# R-PointHop: A Green and High Performance Point Cloud Registration Method

R-PointHop is an unsupervised learning based method for registration of 3D point cloud objcets. This is an official implementation of R-PointHop by [Pranav Kadam](https://github.com/pranavkdm), [Min Zhang](https://github.com/minzhang-1), Shan Liu and C.-C. Jay Kuo. This work was carried out at Media Communications Lab (MCL), University of Southern California, USA.

[[arXiv](https://arxiv.org/)]

## Introduction

In this repository, we release the code for training R-PointHop method and evaluating on a given pair of point cloud objects.

## Packages

The code has been developed and tested in Python 3.6. The following packages need to be installed.

```
h5py
numpy
scipy
sklearn
open3d
```

## Training

Train the model on all 40 classes of ModelNet40 dataset

```
python train.py --first_20 False
```

Train the model on first 20 classes of ModelNet40 dataset

```
python train.py --first_20 True
```

User can specify other parameters like number of points in each hop, neighborhood size and energy threshold, else default parameters will be used.

## Citation

If you find our work useful in your research, please consider citing:

```
BibTeX information coming soon
```
